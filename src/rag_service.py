"""
RAGサービスモジュール

ドキュメント処理、エンベディング生成、ベクトルデータベースを統合して、
インデックス化と検索の機能を提供します。
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Iterable, Tuple, Sequence

from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase


class RAGService:
    """
    RAGサービスクラス

    ドキュメント処理、エンベディング生成、ベクトルデータベースを統合して、
    インデックス化と検索の機能を提供します。

    Attributes:
        document_processor: ドキュメント処理クラスのインスタンス
        embedding_generator: エンベディング生成クラスのインスタンス
        vector_database: ベクトルデータベースクラスのインスタンス
        logger: ロガー
    """

    def __init__(
        self, document_processor: DocumentProcessor, embedding_generator: EmbeddingGenerator, vector_database: VectorDatabase
    ):
        """
        RAGServiceのコンストラクタ

        Args:
            document_processor: ドキュメント処理クラスのインスタンス
            embedding_generator: エンベディング生成クラスのインスタンス
            vector_database: ベクトルデータベースクラスのインスタンス
        """
        # ロガーの設定
        self.logger = logging.getLogger("rag_service")
        self.logger.setLevel(logging.INFO)

        # コンポーネントの設定
        self.document_processor = document_processor
        self.embedding_generator = embedding_generator
        self.vector_database = vector_database

        # データベースの初期化
        try:
            self.vector_database.initialize_database()
            try:
                migrated = self.vector_database.migrate_default_scope_to_library()
                if migrated:
                    self.logger.info(f"defaultスコープのレガシーデータを {migrated} 件 library へ移行しました")
            except Exception as migrate_err:
                self.logger.warning(f"レガシーデータの移行に失敗しました: {migrate_err}")
        except Exception as e:
            self.logger.error(f"データベースの初期化に失敗しました: {str(e)}")
            raise

    @staticmethod
    def _build_doc_base_id(tenant: str, notebook: str, file_name: str) -> str:
        """
        テナントとファイル名から一意なドキュメントIDを生成します。
        """
        safe_tenant = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tenant.strip()) or "default"
        safe_notebook = re.sub(r"[^a-zA-Z0-9_.-]+", "-", notebook.strip()) or "notebook"
        base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", file_name.strip()) or "document"
        return f"{safe_tenant}:{safe_notebook}:{base}"

    @staticmethod
    def _build_chunk_id(doc_base_id: str, chunk_index: int) -> str:
        return f"{doc_base_id}::chunk-{chunk_index}"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        lowered = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9一-龥ぁ-んァ-ヴー]+", lowered)
        return [tok for tok in tokens if len(tok) > 1]

    def _gate_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        limit: int,
        *,
        relax: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        検索候補から利用に足るチャンクを抽出するゲート処理。
        relax=True の場合は閾値を緩めて候補を拾いやすくする。
        Returns:
            (採用された候補, 各候補の判定情報リスト)
        """
        if not results:
            return [], []

        query_tokens = self._tokenize(query)
        selected: List[Dict[str, Any]] = []
        stats: List[Dict[str, Any]] = []

        sim_threshold = self.SIMILARITY_THRESHOLD if not relax else self.FALLBACK_SIMILARITY_THRESHOLD
        min_hits = self.MIN_LEXICAL_HITS if not relax else 0

        for item in results:
            similarity = float(item.get("similarity") or 0.0)
            rerank_score = float(item.get("rerank_score") or similarity)
            content = (item.get("content") or "").lower()
            title = (item.get("title") or "").lower()
            lexical_hit = any(tok in content for tok in query_tokens) if query_tokens else False
            title_hit = any(tok in title for tok in query_tokens) if query_tokens else False

            accepted = similarity >= sim_threshold
            if accepted and min_hits > 0:
                accepted = lexical_hit or title_hit

            if relax and not accepted:
                # 予備条件：類似度が僅差でも rerank が高い、もしくは題名一致があれば採用
                accepted = (
                    similarity >= self.FALLBACK_SIMILARITY_THRESHOLD
                    and (lexical_hit or title_hit or rerank_score >= self.FALLBACK_RERANK_THRESHOLD)
                )

            stats.append(
                {
                    "item": item,
                    "similarity": similarity,
                    "rerank": rerank_score,
                    "lexical": lexical_hit,
                    "title": title_hit,
                    "accepted": bool(accepted),
                    "stage": "relaxed" if relax else "primary",
                    "threshold": sim_threshold,
                }
            )

            if accepted:
                selected.append(item)
                if len(selected) >= self.MAX_RESULTS:
                    break

        if selected:
            return selected[: min(limit, len(selected))], stats

        return [], stats

    def index_documents(
        self,
        source_dir: str,
        processed_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        incremental: bool = False,
        *,
        tenant: str = "default",
        notebook: str = "default",
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        include_global: bool = False,
    ) -> Dict[str, Any]:
        """
        ディレクトリ内のファイルをインデックス化します。

        Args:
            source_dir: インデックス化するファイルが含まれるディレクトリのパス
            processed_dir: 処理済みファイルを保存するディレクトリのパス
            chunk_size: チャンクサイズ（文字数）
            chunk_overlap: チャンク間のオーバーラップ（文字数）
            incremental: 差分のみをインデックス化するかどうか
            tenant: テナントID（ノートブックIDなど）
        """
        start_time = time.time()
        document_count = 0

        base_processed = processed_dir or os.environ.get("PROCESSED_DIR", "data/processed")
        os.makedirs(base_processed, exist_ok=True)

        try:
            user_key = (user_id or "").strip()
            notebook_key = (notebook_id or "").strip()
            if not user_key:
                raise ValueError("user_id が指定されていません")
            if not notebook_key:
                raise ValueError("notebook_id が指定されていません")
            if incremental:
                self.logger.info(
                    f"ディレクトリ '{source_dir}' （tenant={tenant}, notebook={notebook}) の差分ファイルをインデックス化しています..."
                )
            else:
                self.logger.info(
                    f"ディレクトリ '{source_dir}' （tenant={tenant}, notebook={notebook}) のファイルをインデックス化しています..."
                )

            chunks = self.document_processor.process_directory(
                source_dir, base_processed, chunk_size, chunk_overlap, incremental
            )

            if not chunks:
                self.logger.warning(f"ディレクトリ '{source_dir}' 内に処理可能なファイルが見つかりませんでした")
                return {
                    "document_count": 0,
                    "processing_time": time.time() - start_time,
                    "success": True,
                    "message": f"ディレクトリ '{source_dir}' に新規ドキュメントはありませんでした",
                    "doc_ids": [],
                }

            self.logger.info(f"{len(chunks)} チャンクのエンベディングを生成しています...")
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)

            documents = []
            doc_ids = set()
            for i, chunk in enumerate(chunks):
                meta = dict(chunk.get("metadata") or {})
                file_name = meta.get("file_name") or os.path.basename(chunk["file_path"])
                original_path = chunk.get("original_file_path") or meta.get("original_file_path") or chunk["file_path"]
                doc_base_id = self._build_doc_base_id(tenant, notebook, file_name)
                doc_ids.add(doc_base_id)
                metadata = {
                    **meta,
                    "file_name": file_name,
                    "tenant": tenant,
                    "notebook": notebook,
                    "user_id": user_key,
                    "notebook_id": notebook_key,
                    "doc_base_id": doc_base_id,
                    "source_file_path": original_path,
                    "processed_file_path": chunk["file_path"],
                }
                documents.append(
                    {
                        "document_id": self._build_chunk_id(doc_base_id, chunk["chunk_index"]),
                        "tenant": tenant,
                        "notebook": notebook,
                        "user_id": user_key,
                        "notebook_id": notebook_key,
                        "doc_base_id": doc_base_id,
                        "title": file_name,
                        "content": chunk["content"],
                        "file_path": chunk["file_path"],
                        "chunk_index": chunk["chunk_index"],
                        "embedding": embeddings[i],
                        "metadata": {
                            **metadata,
                            "page": meta.get("page"),
                            "source_uri": meta.get("source_uri") or original_path,
                            "txt_uri": meta.get("txt_uri") or chunk["file_path"],
                            "offsets": meta.get("offsets"),
                        },
                    }
                )

            self.vector_database.batch_insert_documents(documents)
            document_count = len(documents)

            processing_time = time.time() - start_time
            self.logger.info(
                f"インデックス化が完了しました（{document_count} チャンク、テナント {tenant} / ノートブック {notebook}、{processing_time:.2f} 秒）"
            )

            return {
                "document_count": document_count,
                "processing_time": processing_time,
                "success": True,
                "message": f"{document_count} チャンクをインデックス化しました",
                "doc_ids": sorted(doc_ids),
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"インデックス化中にエラーが発生しました: {str(e)}")

            return {
                "document_count": document_count,
                "processing_time": processing_time,
                "success": False,
                "error": str(e),
                "doc_ids": [],
            }

    def search(
        self,
        query: str,
        limit: int = 5,
        with_context: bool = False,
        context_size: int = 1,
        full_document: bool = False,
        *,
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        doc_filter: Optional[Iterable[str]] = None,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        include_global: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        ベクトル検索を行います。

        Args:
            query: 検索クエリ
            limit: 返す結果の数（デフォルト: 5）
            with_context: 前後のチャンクも取得するかどうか（デフォルト: False）
            context_size: 前後に取得するチャンク数（デフォルト: 1）
            full_document: ドキュメント全体を取得するかどうか（デフォルト: False）

        Returns:
            検索結果のリスト（関連度順）
                - document_id: ドキュメントID
                - content: コンテンツ
                - file_path: ファイルパス
                - similarity: 類似度
                - metadata: メタデータ
                - is_context: コンテキストチャンクかどうか（前後のチャンクの場合はTrue）
                - is_full_document: 全文ドキュメントかどうか（ドキュメント全体の場合はTrue）
        """
        try:
            # クエリからエンベディングを生成
            self.logger.info(f"クエリ '{query}' のエンベディングを生成しています...")
            query_embedding = self.embedding_generator.generate_search_embedding(query)

            doc_filter_list = list(doc_filter) if doc_filter else None
            user_key = (user_id or "").strip()
            notebook_key = (notebook_id or "").strip()
            if not user_key:
                raise ValueError("search には user_id が必須です")
            if not notebook_key:
                raise ValueError("search には notebook_id が必須です")
            doc_count = self.vector_database.get_document_count(
                tenant,
                notebook,
                user_id=user_key,
                notebook_id=notebook_key,
                include_global=include_global,
            )
            if doc_count == 0:
                self.logger.info("対象テナントにドキュメントが存在しません")
                return []

            self.logger.info(
                f"クエリ '{query}' でベクトル検索を実行しています... (tenant={tenant or 'default'}, notebook={notebook or 'default'}, limit={limit})"
            )
            search_limit = max(limit, self.PRIMARY_SEARCH_CANDIDATES)
            results = self.vector_database.search(
                query_embedding,
                search_limit,
                tenant=tenant,
                notebook=notebook,
                doc_filter=doc_filter_list,
                user_id=user_key,
                notebook_id=notebook_key,
                include_global=include_global,
            )
            keyword_terms = self._extract_priority_keywords(query)
            if keyword_terms:
                keyword_hits = self.vector_database.search_by_keywords(
                    keyword_terms,
                    limit=self.KEYWORD_SEARCH_LIMIT,
                    tenant=tenant,
                    notebook=notebook,
                    user_id=user_key,
                    notebook_id=notebook_key,
                    include_global=include_global,
                )
                if keyword_hits:
                    keyword_hits = self._prepare_keyword_hits(keyword_hits, keyword_terms)
                    results = self._merge_keyword_hits(results, keyword_hits, search_limit * 2)
            gated, gate_stats = self._gate_results(query, results, limit)

            def _annotate_gate(stats: List[Dict[str, Any]]) -> None:
                for st in stats:
                    item = st["item"]
                    item["_gate_passed"] = bool(st["accepted"])
                    item["_gate_stage"] = st["stage"]
                    item["_gate_similarity"] = st["similarity"]
                    item["_gate_rerank"] = st["rerank"]
                    item["_gate_lexical"] = st["lexical"]
                    item["_gate_title"] = st["title"]
                    item["_gate_threshold"] = st["threshold"]
                    meta = dict(item.get("metadata") or {})
                    if "is_global" not in meta:
                        meta["is_global"] = bool(item.get("is_global"))
                    meta["gate"] = {
                        "accepted": bool(st["accepted"]),
                        "stage": st["stage"],
                        "similarity": st["similarity"],
                        "rerank": st["rerank"],
                        "lexical": st["lexical"],
                        "title": st["title"],
                        "threshold": st["threshold"],
                    }
                    item["metadata"] = meta

            _annotate_gate(gate_stats)

            if not gated:
                expanded_limit = min(
                    self.FALLBACK_SEARCH_MAX,
                    max(int(limit * self.FALLBACK_SEARCH_MULTIPLIER), search_limit, limit),
                )
                if expanded_limit > search_limit:
                    self.logger.info(
                        f"ゲート通過0件のため候補を拡張検索します (limit={expanded_limit})"
                    )
                    results = self.vector_database.search(
                        query_embedding,
                        expanded_limit,
                        tenant=tenant,
                        notebook=notebook,
                        doc_filter=doc_filter_list,
                        user_id=user_key,
                        notebook_id=notebook_key,
                        include_global=include_global,
                    )
                    gated, gate_stats = self._gate_results(query, results, limit)
                    _annotate_gate(gate_stats)

            if not gated:
                # 緩和条件でもう一度判定
                relaxed, relaxed_stats = self._gate_results(query, results, limit, relax=True)
                _annotate_gate(relaxed_stats)
                if relaxed:
                    gated = relaxed
                    gate_stats = relaxed_stats

            if not gated:
                # 最終フォールバック：上位チャンクをそのまま提示（ゲート不合格）
                fallback_limit = min(limit, len(results))
                gated = results[:fallback_limit]
                fallback_stats: List[Dict[str, Any]] = []
                for item in gated:
                    item["_gate_passed"] = False
                    item["_gate_stage"] = "fallback"
                    item["_gate_similarity"] = float(item.get("similarity") or 0.0)
                    item["_gate_rerank"] = float(item.get("rerank_score") or item.get("similarity") or 0.0)
                    item["_gate_lexical"] = False
                    item["_gate_title"] = False
                    item["_gate_threshold"] = self.FALLBACK_SIMILARITY_THRESHOLD
                    meta = dict(item.get("metadata") or {})
                    meta["gate"] = {
                        "accepted": False,
                        "stage": "fallback",
                        "similarity": item["_gate_similarity"],
                        "rerank": item["_gate_rerank"],
                        "lexical": False,
                        "title": False,
                        "threshold": self.FALLBACK_SIMILARITY_THRESHOLD,
                    }
                    item["metadata"] = meta
                    fallback_stats.append(
                        {
                            "item": item,
                            "similarity": item["_gate_similarity"],
                            "rerank": item["_gate_rerank"],
                            "lexical": False,
                            "title": False,
                            "accepted": False,
                            "stage": "fallback",
                            "threshold": self.FALLBACK_SIMILARITY_THRESHOLD,
                        }
                    )
                gate_stats = fallback_stats

            passed = sum(1 for item in gated if item.get("_gate_passed"))
            self.logger.info(
                f"ゲート通過: {passed}/{len(gated)} 件（tenant={tenant}, notebook={notebook}）"
            )

            # 前後のチャンクも取得する場合
            if with_context and context_size > 0:
                context_results = []
                processed_files = set()  # 処理済みのファイルとチャンクの組み合わせを記録

                for result in gated:
                    file_path = result["file_path"]
                    chunk_index = result["chunk_index"]
                    file_chunk_key = f"{file_path}_{chunk_index}"

                    # 既に処理済みのファイルとチャンクの組み合わせはスキップ
                    if file_chunk_key in processed_files:
                        continue

                    processed_files.add(file_chunk_key)

                    # 前後のチャンクを取得
                    adjacent_chunks = self.vector_database.get_adjacent_chunks(file_path, chunk_index, context_size)
                    context_results.extend(adjacent_chunks)

                # 結果をマージ
                all_results = gated.copy()

                # 重複を避けるために、既に結果に含まれているドキュメントIDを記録
                existing_doc_ids = {result["document_id"] for result in all_results}

                # 重複していないコンテキストチャンクのみを追加
                for context in context_results:
                    if context["document_id"] not in existing_doc_ids:
                        all_results.append(context)
                        existing_doc_ids.add(context["document_id"])

                # ファイルパスとチャンクインデックスでソート
                all_results.sort(key=lambda x: (x["file_path"], x["chunk_index"]))

                self.logger.info(f"検索結果（コンテキスト含む）: {len(all_results)} 件")

                # ドキュメント全体を取得する場合
                if full_document:
                    full_doc_results = []
                    processed_files = set()  # 処理済みのファイルを記録

                    # 検索結果に含まれるファイルの全文を取得
                    for result in all_results:
                        file_path = result["file_path"]

                        # 既に処理済みのファイルはスキップ
                        if file_path in processed_files:
                            continue

                        processed_files.add(file_path)

                        # ファイルの全文を取得
                        full_doc_chunks = self.vector_database.get_document_by_file_path(file_path)
                        full_doc_results.extend(full_doc_chunks)

                    # 結果をマージ
                    merged_results = all_results.copy()

                    # 重複を避けるために、既に結果に含まれているドキュメントIDを記録
                    existing_doc_ids = {result["document_id"] for result in merged_results}

                    # 重複していない全文チャンクのみを追加
                    for doc_chunk in full_doc_results:
                        if doc_chunk["document_id"] not in existing_doc_ids:
                            merged_results.append(doc_chunk)
                            existing_doc_ids.add(doc_chunk["document_id"])

                    # ファイルパスとチャンクインデックスでソート
                    merged_results.sort(key=lambda x: (x["file_path"], x["chunk_index"]))

                    self.logger.info(f"検索結果（全文含む）: {len(merged_results)} 件")
                    return merged_results
                else:
                    return all_results
            else:
                # ドキュメント全体を取得する場合
                if full_document:
                    full_doc_results = []
                    processed_files = set()  # 処理済みのファイルを記録

                    # 検索結果に含まれるファイルの全文を取得
                    for result in gated:
                        file_path = result["file_path"]

                        # 既に処理済みのファイルはスキップ
                        if file_path in processed_files:
                            continue

                        processed_files.add(file_path)

                        # ファイルの全文を取得
                        full_doc_chunks = self.vector_database.get_document_by_file_path(file_path)
                        full_doc_results.extend(full_doc_chunks)

                    # 結果をマージ
                    merged_results = gated.copy()

                    # 重複を避けるために、既に結果に含まれているドキュメントIDを記録
                    existing_doc_ids = {result["document_id"] for result in merged_results}

                    # 重複していない全文チャンクのみを追加
                    for doc_chunk in full_doc_results:
                        if doc_chunk["document_id"] not in existing_doc_ids:
                            merged_results.append(doc_chunk)
                            existing_doc_ids.add(doc_chunk["document_id"])

                    # ファイルパスとチャンクインデックスでソート
                    merged_results.sort(key=lambda x: (x["file_path"], x["chunk_index"]))

                    self.logger.info(f"検索結果（全文含む）: {len(merged_results)} 件")
                    return merged_results
                else:
                    self.logger.info(f"検索結果: {len(gated)} 件")
                    return gated

        except Exception as e:
            self.logger.error(f"検索中にエラーが発生しました: {str(e)}")
            raise

    def _extract_priority_keywords(self, query: str) -> List[str]:
        lowered = (query or "").lower()
        hits: List[str] = []
        for term in self.PRIORITY_KEYWORD_TERMS:
            if term.lower() in lowered:
                hits.append(term)
        return hits

    def _keyword_matches_text(self, text: str, keywords: Sequence[str]) -> List[str]:
        if not text:
            return []
        lowered = text.lower()
        return [kw for kw in keywords if kw.lower() in lowered]

    def _prepare_keyword_hits(
        self,
        hits: List[Dict[str, Any]],
        keywords: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        for idx, hit in enumerate(hits):
            matches = self._keyword_matches_text(hit.get("content") or "", keywords)
            meta = dict(hit.get("metadata") or {})
            meta.setdefault("keyword_match", {})
            meta["keyword_match"] = {
                "query_keywords": list(keywords),
                "matches": matches,
            }
            hit["metadata"] = meta
            base = float(hit.get("similarity") or 0.0)
            boosted = max(base, self.KEYWORD_SIMILARITY_BASE) + max(
                0.0, self.KEYWORD_SIMILARITY_BONUS - idx * 0.02
            )
            hit["similarity"] = min(0.999, boosted)
            hit["_keyword_priority"] = True
        return hits

    def _merge_keyword_hits(
        self,
        base_results: List[Dict[str, Any]],
        keyword_hits: List[Dict[str, Any]],
        max_candidates: int,
    ) -> List[Dict[str, Any]]:
        if not keyword_hits:
            return base_results
        merged = list(base_results)
        seen_ids = {item["document_id"] for item in merged if item.get("document_id")}
        for hit in keyword_hits:
            doc_id = hit.get("document_id")
            if doc_id in seen_ids:
                continue
            merged.append(hit)
            if doc_id:
                seen_ids.add(doc_id)
            if len(merged) >= max_candidates:
                break
        return merged

    def clear_index(self) -> Dict[str, Any]:
        """
        インデックスをクリアします。

        Returns:
            クリアの結果
                - deleted_count: 削除されたドキュメント数
                - success: 成功したかどうか
                - error: エラーメッセージ（エラーが発生した場合）
        """
        try:
            # データベースをクリア
            self.logger.info("インデックスをクリアしています...")
            deleted_count = self.vector_database.clear_database()

            self.logger.info(f"インデックスをクリアしました（{deleted_count} ドキュメントを削除）")
            return {"deleted_count": deleted_count, "success": True, "message": f"{deleted_count} ドキュメントを削除しました"}

        except Exception as e:
            self.logger.error(f"インデックスのクリア中にエラーが発生しました: {str(e)}")

            return {"deleted_count": 0, "success": False, "error": str(e)}

    def get_document_count(
        self,
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        include_global: bool = False,
    ) -> int:
        """
        インデックス内のドキュメント数を取得します。
        """
        try:
            user_key = (user_id or "").strip()
            notebook_key = (notebook_id or "").strip()
            if not user_key or not notebook_key:
                raise ValueError("user_id と notebook_id は必須です")
            scoped_notebook = (notebook or notebook_key).strip() or notebook_key
            count = self.vector_database.get_document_count(
                tenant,
                scoped_notebook,
                user_id=user_key,
                notebook_id=notebook_key,
                include_global=include_global,
            )
            self.logger.info(f"インデックス内のドキュメント数(tenant={tenant}, notebook={notebook}): {count}")
            return count
        except Exception as e:
            self.logger.error(f"ドキュメント数の取得中にエラーが発生しました: {str(e)}")
            raise

    def list_documents(
        self,
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        include_global: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        ドキュメント一覧を取得します。
        """
        try:
            user_key = (user_id or "").strip()
            notebook_key = (notebook_id or "").strip()
            if not user_key or not notebook_key:
                raise ValueError("user_id と notebook_id は必須です")
            scoped_notebook = (notebook or notebook_key).strip() or notebook_key
            docs = self.vector_database.list_documents(
                tenant,
                scoped_notebook,
                user_id=user_key,
                notebook_id=notebook_key,
                include_global=include_global,
            )
            self.logger.info(f"ドキュメント一覧を取得しました（{len(docs)} 件, tenant={tenant}, notebook={notebook}）")
            return docs
        except Exception as e:
            self.logger.error(f"ドキュメント一覧の取得中にエラーが発生しました: {str(e)}")
            raise

    def list_notebooks(
        self,
        *,
        user_id: str,
        tenant: Optional[str] = None,
        include_global: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        ノートブック単位の集計を取得します。
        """
        user_key = (user_id or "").strip()
        if not user_key:
            raise ValueError("user_id は必須です")
        return self.vector_database.list_notebook_summaries(
            user_id=user_key,
            tenant=tenant,
            include_global=include_global,
        )

    def delete_document(self, tenant: str, notebook: str, doc_base_id: str) -> int:
        """
        指定ドキュメントを削除します。
        """
        deleted = self.vector_database.delete_document_by_base_id(tenant, notebook, doc_base_id)
        return deleted

    def delete_tenant_documents(self, tenant: str, notebook: Optional[str] = None) -> int:
        """
        テナント全体のドキュメントを削除します。
        """
        return self.vector_database.delete_documents_by_tenant(tenant, notebook)
 
    def delete_notebook(
        self,
        *,
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        user_id: str,
        notebook_id: str,
    ) -> int:
        """
        指定したユーザー・ノートブックのドキュメントを削除します。
        """
        user_key = (user_id or "").strip()
        notebook_key = (notebook_id or "").strip()
        if not user_key or not notebook_key:
            raise ValueError("ノートブック削除には user_id と notebook_id が必須です")
        scoped_notebook = (notebook or notebook_key).strip() or notebook_key
        return self.vector_database.delete_documents_by_scope(
            tenant=tenant,
            notebook=scoped_notebook,
            user_id=user_key,
            notebook_id=notebook_key,
        )

    def get_document_metadata(
        self,
        tenant: str,
        notebook: str,
        doc_base_id: str,
        *,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        ドキュメントのメタデータを取得します。
        """
        user_key = (user_id or "").strip()
        notebook_key = (notebook_id or "").strip()
        if not user_key or not notebook_key:
            raise ValueError("ドキュメントメタデータ取得には user_id と notebook_id が必須です")
        scoped_notebook = (notebook or notebook_key).strip() or notebook_key
        return self.vector_database.get_document_metadata(
            tenant,
            scoped_notebook,
            doc_base_id,
            user_id=user_key,
            notebook_id=notebook_key,
        )

    def get_document_overview(
        self,
        doc_base_id: str,
        *,
        tenant: str,
        notebook: str,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        プレビュー表示用のメタ情報を取得し、アクセス制御を確認する。
        """
        user_key = (user_id or "").strip()
        notebook_key = (notebook_id or "").strip()
        if not user_key or not notebook_key:
            raise ValueError("ドキュメント参照には user_id と notebook_id が必須です")
        scoped_notebook = (notebook or notebook_key).strip() or notebook_key
        meta = self.vector_database.get_document_metadata(
            tenant,
            scoped_notebook,
            doc_base_id,
            user_id=user_key,
            notebook_id=notebook_key,
        )
        if not meta:
            raise ValueError("document_not_found")

        stored_user = (meta.get("user_id") or "").strip()
        stored_notebook = (meta.get("notebook_id") or "").strip()
        if user_id and stored_user and stored_user != user_id:
            raise PermissionError("forbidden")
        if notebook_id and stored_notebook and stored_notebook != notebook_id:
            raise PermissionError("forbidden")

        source_uri = meta.get("source_uri") or meta.get("source_file_path")
        txt_uri = meta.get("txt_uri") or meta.get("processed_file_path")
        page_count = meta.get("page_count") or None

        import mimetypes

        mime, _ = mimetypes.guess_type(source_uri or "")

        return {
            "doc_id": doc_base_id,
            "title": meta.get("title") or doc_base_id,
            "source_uri": source_uri,
            "source_file_path": meta.get("source_file_path"),
            "processed_file_path": meta.get("processed_file_path"),
            "txt_uri": txt_uri,
            "page_count": page_count,
            "mime": mime or meta.get("mime") or "application/octet-stream",
            "tenant": tenant,
            "notebook": notebook,
            "user_id": stored_user or user_id,
            "notebook_id": stored_notebook or notebook_id,
        }

    def get_document_chunks(
        self,
        *,
        doc_ids: List[str],
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        user_id: str,
        notebook_id: str,
        include_global: bool = False,
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        指定した doc_ids に紐づくチャンクをまとめて取得します。
        """
        user_key = (user_id or "").strip()
        notebook_key = (notebook_id or "").strip()
        if not user_key or not notebook_key:
            raise ValueError("user_id と notebook_id は必須です")
        scoped_notebook = (notebook or notebook_key).strip() or notebook_key
        return self.vector_database.get_chunks_for_documents(
            doc_ids=doc_ids,
            tenant=tenant,
            notebook=scoped_notebook,
            user_id=user_key,
            notebook_id=notebook_key,
            include_global=include_global,
            max_chunks=max_chunks,
        )

    def warm_up(
        self,
        tenant: str,
        notebook: str,
        *,
        user_id: str,
        notebook_id: str,
        top_k: int = 3,
    ) -> None:
        """
        埋め込みモデルとベクトルデータベースをウォームアップし、初回クエリの遅延を抑える。
        """
        try:
            self.logger.info(
                f"ウォームアップを開始します (tenant={tenant}, notebook={notebook}, top_k={top_k})"
            )
            # 埋め込みモデルのロードをウォームアップ
            self.embedding_generator.generate_search_embedding("warmup")
            user_key = (user_id or "").strip()
            notebook_key = (notebook_id or "").strip()
            if not user_key or not notebook_key:
                self.logger.info("ウォームアップ用のスコープ情報が不足しているためスキップします")
                return
            scoped_notebook = (notebook or notebook_key).strip() or notebook_key
            if self.vector_database.get_document_count(
                tenant,
                scoped_notebook,
                user_id=user_key,
                notebook_id=notebook_key,
            ) == 0:
                self.logger.info("対象ドキュメントが存在しないためウォームアップをスキップします")
                return
            # ベクトルDBに対して軽い検索を実行
            self.search(
                "warmup",
                limit=max(1, top_k),
                tenant=tenant,
                notebook=scoped_notebook,
                user_id=user_key,
                notebook_id=notebook_key,
            )
            self.logger.info("ウォームアップが完了しました")
        except Exception as exc:
            self.logger.warning(f"ウォームアップに失敗しました: {exc}")
    SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIM_THRESHOLD", "0.35"))
    MIN_LEXICAL_HITS = int(os.getenv("RETRIEVAL_MIN_LEXICAL_HITS", "1"))
    MAX_RESULTS = int(os.getenv("RETRIEVAL_MAX_RESULTS", "40"))
    FALLBACK_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_FALLBACK_SIM_THRESHOLD", "0.25"))
    FALLBACK_RERANK_THRESHOLD = float(os.getenv("RETRIEVAL_FALLBACK_RERANK_THRESHOLD", "0.45"))
    FALLBACK_SEARCH_MULTIPLIER = float(os.getenv("RETRIEVAL_FALLBACK_SEARCH_MULTIPLIER", "2.0"))
    FALLBACK_SEARCH_MAX = int(os.getenv("RETRIEVAL_FALLBACK_SEARCH_MAX", "50"))
    PRIMARY_SEARCH_CANDIDATES = int(os.getenv("RETRIEVAL_PRIMARY_CANDIDATES", "20"))
    PRIORITY_KEYWORD_TERMS = tuple(
        term.strip()
        for term in os.getenv("RETRIEVAL_PRIORITY_KEYWORDS", "ゼロショットプロンプト,ゼロショット,zero-shot,zero shot").split(",")
        if term.strip()
    )
    KEYWORD_SEARCH_LIMIT = int(os.getenv("RETRIEVAL_KEYWORD_SEARCH_LIMIT", "10"))
    KEYWORD_SIMILARITY_BASE = float(os.getenv("RETRIEVAL_KEYWORD_SIM_BASE", "0.72"))
    KEYWORD_SIMILARITY_BONUS = float(os.getenv("RETRIEVAL_KEYWORD_SIM_BONUS", "0.18"))
