"""
ベクトルデータベースモジュール

PostgreSQLとpgvectorを使用してベクトルの保存と検索を行います。
"""

import logging
import psycopg2
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Sequence

# .envの読み込み
load_dotenv()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


class VectorDatabase:
    """
    ベクトルデータベースクラス

    PostgreSQLとpgvectorを使用してベクトルの保存と検索を行います。

    Attributes:
        connection_params: 接続パラメータ
        connection: データベース接続
        logger: ロガー
    """

    def __init__(self, connection_params: Dict[str, Any]):
        """
        VectorDatabaseのコンストラクタ

        Args:
            connection_params: 接続パラメータ
                - host: ホスト名
                - port: ポート番号
                - user: ユーザー名
                - password: パスワード
                - database: データベース名
        """
        # ロガーの設定
        self.logger = logging.getLogger("vector_database")
        self.logger.setLevel(logging.INFO)

        # 接続パラメータの保存
        self.connection_params = connection_params
        self.connection = None

    def connect(self) -> None:
        """
        データベースに接続します。

        Raises:
            Exception: 接続に失敗した場合
        """
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.logger.info("データベースに接続しました")
        except Exception as e:
            self.logger.error(f"データベースへの接続に失敗しました: {str(e)}")
            raise

    def disconnect(self) -> None:
        """
        データベースから切断します。
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("データベースから切断しました")

    def _ensure_connection(self) -> None:
        """接続が閉じている場合は再接続する。"""
        if self.connection is None:
            self.connect()
            return

        if getattr(self.connection, "closed", 1):
            self.logger.warning("データベース接続がクローズされていたため再接続します")
            self.connection = None
            self.connect()

    def _safe_rollback(self) -> None:
        """ロールバック前に接続状態を確認する。"""
        if self.connection is None or getattr(self.connection, "closed", 1):
            return

        try:
            self.connection.rollback()
        except Exception as rollback_error:
            self.logger.warning(f"ロールバックに失敗しました: {rollback_error}")

    def initialize_database(self) -> None:
        """
        データベースを初期化します。

        テーブルとインデックスを作成します。

        Raises:
            Exception: 初期化に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            schema_statements = [
                "CREATE EXTENSION IF NOT EXISTS vector;",
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT UNIQUE NOT NULL,
                    tenant TEXT NOT NULL DEFAULT 'default',
                    notebook TEXT NOT NULL DEFAULT '',
                    doc_base_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata JSONB,
                    embedding vector({EMBEDDING_DIM}),
                    user_id TEXT NOT NULL DEFAULT 'default',
                    notebook_id TEXT NOT NULL DEFAULT 'default',
                    is_global BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """,
                """
                ALTER TABLE documents
                    ADD COLUMN IF NOT EXISTS tenant TEXT NOT NULL DEFAULT 'default',
                    ADD COLUMN IF NOT EXISTS notebook TEXT NOT NULL DEFAULT '',
                    ADD COLUMN IF NOT EXISTS doc_base_id TEXT NOT NULL DEFAULT '',
                    ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT '',
                    ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'default',
                    ADD COLUMN IF NOT EXISTS notebook_id TEXT NOT NULL DEFAULT 'default',
                    ADD COLUMN IF NOT EXISTS is_global BOOLEAN NOT NULL DEFAULT FALSE;
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents (document_id);
                CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents (file_path);
                CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents (tenant);
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_notebook ON documents (tenant, notebook);
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_doc ON documents (tenant, doc_base_id);
                CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents (user_id, notebook_id);
                CREATE INDEX IF NOT EXISTS idx_documents_global ON documents (is_global);
                CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """,
                "ANALYZE documents;",
            ]

            for statement in schema_statements:
                cursor.execute(statement)

            # コミット
            self.connection.commit()
            self.logger.info("データベースを初期化しました")

        except Exception as e:
            # ロールバック
            self._safe_rollback()
            self.logger.error(f"データベースの初期化に失敗しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def insert_document(
        self,
        document_id: str,
        content: str,
        file_path: str,
        chunk_index: int,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        *,
        tenant: str = "default",
        notebook: str = "",
        doc_base_id: str = "",
        title: str = "",
        user_id: str = "default",
        notebook_id: str = "default",
        is_global: bool = False,
    ) -> None:
        """
        ドキュメントを挿入します。

        Args:
            document_id: ドキュメントID
            content: ドキュメントの内容
            file_path: ファイルパス
            chunk_index: チャンクインデックス
            embedding: エンベディング
            metadata: メタデータ（オプション）

        Raises:
            Exception: 挿入に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # メタデータをJSON形式に変換
            metadata_json = json.dumps(metadata) if metadata else None

            # ドキュメントの挿入
            cursor.execute(
                """
                INSERT INTO documents (document_id, tenant, notebook, doc_base_id, title, content, file_path, chunk_index, embedding, metadata, user_id, notebook_id, is_global)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) 
                DO UPDATE SET 
                    tenant = EXCLUDED.tenant,
                    notebook = EXCLUDED.notebook,
                    doc_base_id = EXCLUDED.doc_base_id,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    file_path = EXCLUDED.file_path,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    user_id = EXCLUDED.user_id,
                    notebook_id = EXCLUDED.notebook_id,
                    is_global = EXCLUDED.is_global,
                    created_at = CURRENT_TIMESTAMP;
            """,
                (
                    document_id,
                    tenant,
                    notebook,
                    doc_base_id,
                    title,
                    content,
                    file_path,
                    chunk_index,
                    embedding,
                    metadata_json,
                    user_id,
                    notebook_id,
                    bool(is_global),
                ),
            )

            # コミット
            self.connection.commit()
            self.logger.debug(f"ドキュメント '{document_id}' を挿入しました")

        except Exception as e:
            # ロールバック
            self._safe_rollback()
            self.logger.error(f"ドキュメントの挿入に失敗しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def batch_insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        複数のドキュメントをバッチ挿入します。

        Args:
            documents: ドキュメントのリスト
                各ドキュメントは以下のキーを持つ辞書:
                - document_id: ドキュメントID
                - content: ドキュメントの内容
                - file_path: ファイルパス
                - chunk_index: チャンクインデックス
                - embedding: エンベディング
                - metadata: メタデータ（オプション）

        Raises:
            Exception: 挿入に失敗した場合
        """
        if not documents:
            self.logger.warning("挿入するドキュメントがありません")
            return

        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # バッチ挿入用のデータ作成
            values = []
            for doc in documents:
                metadata_json = json.dumps(doc.get("metadata")) if doc.get("metadata") else None
                values.append(
                    (
                        doc["document_id"],
                        doc.get("tenant", "default"),
                        doc.get("notebook", ""),
                        doc.get("doc_base_id", ""),
                        doc.get("title", doc.get("metadata", {}).get("file_name", "")),
                        doc["content"],
                        doc["file_path"],
                        doc["chunk_index"],
                        doc["embedding"],
                        metadata_json,
                        doc.get("user_id", doc.get("tenant", "default")),
                        doc.get("notebook_id", doc.get("notebook", "default")),
                        bool(doc.get("is_global", False)),
                    )
                )

            cursor.executemany(
                """
                INSERT INTO documents (document_id, tenant, notebook, doc_base_id, title, content, file_path, chunk_index, embedding, metadata, user_id, notebook_id, is_global)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) 
                DO UPDATE SET 
                    tenant = EXCLUDED.tenant,
                    notebook = EXCLUDED.notebook,
                    doc_base_id = EXCLUDED.doc_base_id,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    file_path = EXCLUDED.file_path,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    user_id = EXCLUDED.user_id,
                    notebook_id = EXCLUDED.notebook_id,
                    is_global = EXCLUDED.is_global,
                    created_at = CURRENT_TIMESTAMP;
            """,
                values,
            )

            # コミット
            self.connection.commit()
            self.logger.info(f"{len(documents)} 個のドキュメントを挿入しました")

        except Exception as e:
            # ロールバック
            self._safe_rollback()
            self.logger.error(f"ドキュメントのバッチ挿入に失敗しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        doc_filter: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        include_global: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        ベクトル検索を行います。

        Args:
            query_embedding: クエリのエンベディング
            limit: 返す結果の数（デフォルト: 5）

        Returns:
            検索結果のリスト（関連度順）

        Raises:
            Exception: 検索に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # クエリエンベディングをPostgreSQLの配列構文に変換
            embedding_str = str(query_embedding)
            embedding_array = f"ARRAY{embedding_str}::vector"

            # ベクトル検索
            where_clauses = ["embedding IS NOT NULL"]
            params: List[Any] = []
            scope_user = (user_id or "").strip()
            scope_notebook = (notebook_id or "").strip()
            if not scope_user or not scope_notebook:
                raise ValueError("user_id と notebook_id は必須です")
            if tenant:
                where_clauses.append("tenant = %s")
                params.append(tenant)
            if notebook:
                where_clauses.append("notebook = %s")
                params.append(notebook)
            scope_clause = "(user_id = %s AND notebook_id = %s)"
            if include_global:
                scope_clause = f"({scope_clause} OR is_global = TRUE)"
            where_clauses.append(scope_clause)
            params.extend([scope_user, scope_notebook])
            if doc_filter:
                safe_filter = doc_filter
                prefix = None
                if tenant and notebook:
                    prefix = f"{tenant}:{notebook}:"
                elif tenant:
                    prefix = f"{tenant}:"
                if prefix:
                    safe_filter = [d for d in doc_filter if isinstance(d, str) and d.startswith(prefix)]
                else:
                    safe_filter = []
                if safe_filter:
                    placeholders = ", ".join(["%s"] * len(safe_filter))
                    where_clauses.append(f"doc_base_id IN ({placeholders})")
                    params.extend(safe_filter)

            where_sql = " AND ".join(where_clauses)

            query = f"""
                SELECT
                    document_id,
                    tenant,
                    notebook,
                    doc_base_id,
                    title,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    user_id,
                    notebook_id,
                    is_global,
                    1 - (embedding <=> {embedding_array}) AS similarity
                FROM
                    documents
                WHERE
                    {where_sql}
                ORDER BY
                    embedding <=> {embedding_array}
                LIMIT %s;
            """

            params.append(limit)
            cursor.execute(query, params)

            rows = cursor.fetchall()
            results = [self._map_row_to_document(row) for row in rows]

            self.logger.info(f"クエリに対して {len(results)} 件の結果が見つかりました")
            return results

        except Exception as e:
            self.logger.error(f"ベクトル検索中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def search_by_keywords(
        self,
        keywords: Sequence[str],
        limit: int = 5,
        *,
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        include_global: bool = False,
    ) -> List[Dict[str, Any]]:
        """キーワード（部分一致）でチャンクを検索します。"""

        normalized = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
        if not normalized:
            return []

        try:
            self._ensure_connection()

            cursor = self.connection.cursor()

            where_clauses = ["content IS NOT NULL"]
            params: List[Any] = []
            scope_user = (user_id or "").strip()
            scope_notebook = (notebook_id or "").strip()
            if not scope_user or not scope_notebook:
                raise ValueError("user_id と notebook_id は必須です")
            if tenant:
                where_clauses.append("tenant = %s")
                params.append(tenant)
            if notebook:
                where_clauses.append("notebook = %s")
                params.append(notebook)
            scope_clause = "(user_id = %s AND notebook_id = %s)"
            if include_global:
                scope_clause = f"({scope_clause} OR is_global = TRUE)"
            where_clauses.append(scope_clause)
            params.extend([scope_user, scope_notebook])

            keyword_clauses = []
            keyword_score_parts = []
            for kw in normalized:
                pattern = f"%{kw}%"
                keyword_clauses.append("(content ILIKE %s OR title ILIKE %s)")
                params.extend([pattern, pattern])
                keyword_score_parts.append("(CASE WHEN content ILIKE %s THEN 1 ELSE 0 END)")
                keyword_score_parts.append("(CASE WHEN title ILIKE %s THEN 0.5 ELSE 0 END)")
                params.extend([pattern, pattern])

            where_clauses.append(f"({' OR '.join(keyword_clauses)})")
            keyword_score_sql = " + ".join(keyword_score_parts) or "0"

            where_sql = " AND ".join(where_clauses)
            query = f"""
                SELECT
                    document_id,
                    tenant,
                    notebook,
                    doc_base_id,
                    title,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    user_id,
                    notebook_id,
                    is_global,
                    NULL::float8 AS similarity,
                    {keyword_score_sql} AS keyword_hits
                FROM
                    documents
                WHERE
                    {where_sql}
                ORDER BY
                    keyword_hits DESC,
                    created_at DESC
                LIMIT %s;
            """
            params.append(limit)
            cursor.execute(query, params)

            rows = cursor.fetchall()
            results = []
            for row in rows:
                mapped = self._map_row_to_document(row[:13])
                mapped["similarity"] = mapped.get("similarity") or 0.0
                meta = dict(mapped.get("metadata") or {})
                meta["keyword_hits"] = row[13]
                mapped["metadata"] = meta
                results.append(mapped)
            return results
        except Exception as e:
            self.logger.error(f"キーワード検索中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

    def _map_row_to_document(self, row: Sequence[Any]) -> Dict[str, Any]:
        (
            document_id,
            row_tenant,
            row_notebook,
            doc_base_id,
            title,
            content,
            file_path,
            chunk_index,
            metadata_json,
            row_user_id,
            row_notebook_id,
            row_is_global,
            similarity,
        ) = row

        if metadata_json:
            if isinstance(metadata_json, str):
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = metadata_json
        else:
            metadata = {}

        return {
            "document_id": document_id,
            "tenant": row_tenant,
            "notebook": row_notebook,
            "doc_base_id": doc_base_id,
            "title": title,
            "content": content,
            "file_path": file_path,
            "chunk_index": chunk_index,
            "metadata": metadata,
            "user_id": row_user_id,
            "notebook_id": row_notebook_id,
            "is_global": bool(row_is_global),
            "similarity": similarity,
        }

    def delete_document(self, document_id: str) -> bool:
        """
        ドキュメントを削除します。

        Args:
            document_id: 削除するドキュメントのID

        Returns:
            削除に成功した場合はTrue、ドキュメントが見つからない場合はFalse

        Raises:
            Exception: 削除に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ドキュメントの削除
            cursor.execute("DELETE FROM documents WHERE document_id = %s;", (document_id,))

            # 削除された行数を取得
            deleted_rows = cursor.rowcount

            # コミット
            self.connection.commit()

            if deleted_rows > 0:
                self.logger.info(f"ドキュメント '{document_id}' を削除しました")
                return True
            else:
                self.logger.warning(f"ドキュメント '{document_id}' が見つかりません")
                return False

        except Exception as e:
            # ロールバック
            self._safe_rollback()
            self.logger.error(f"ドキュメントの削除中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def delete_by_file_path(self, file_path: str) -> int:
        """
        ファイルパスに基づいてドキュメントを削除します。

        Args:
            file_path: 削除するドキュメントのファイルパス

        Returns:
            削除されたドキュメントの数

        Raises:
            Exception: 削除に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ドキュメントの削除
            cursor.execute("DELETE FROM documents WHERE file_path = %s;", (file_path,))

            # 削除された行数を取得
            deleted_rows = cursor.rowcount

            # コミット
            self.connection.commit()

            self.logger.info(f"ファイルパス '{file_path}' に関連する {deleted_rows} 個のドキュメントを削除しました")
            return deleted_rows

        except Exception as e:
            # ロールバック
            self._safe_rollback()
            self.logger.error(f"ドキュメントの削除中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def clear_database(self) -> int:
        """
        データベースをクリアします（全てのドキュメントを削除）。

        Raises:
            Exception: クリアに失敗した場合

        Returns:
            削除されたドキュメントの数。テーブルをDROPするため、削除前の数を返します。
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # 削除前のドキュメント数を取得
            count_before_delete = self.get_document_count()

            # カーソルの作成
            cursor = self.connection.cursor()

            # テーブルを削除してスキーマもクリア
            cursor.execute("DROP TABLE IF EXISTS documents;")

            # コミット
            self.connection.commit()

            if count_before_delete > 0:
                self.logger.info(
                    f"データベースをクリアしました（documentsテーブルを削除、{count_before_delete} 個のドキュメントが対象でした）"
                )
            else:
                self.logger.info("データベースをクリアしました（documentsテーブルを削除）")
            return count_before_delete

        except Exception as e:
            # ロールバック
            self._safe_rollback()
            self.logger.error(f"データベースのクリア中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

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
        データベース内のドキュメント数を取得します。

        Returns:
            ドキュメント数

        Raises:
            Exception: 取得に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ドキュメント数を取得
            where = []
            params: List[Any] = []
            if tenant:
                where.append("tenant = %s")
                params.append(tenant)
            if notebook:
                where.append("notebook = %s")
                params.append(notebook)
            scope_user = (user_id or "").strip()
            scope_notebook = (notebook_id or "").strip()
            if not scope_user or not scope_notebook:
                raise ValueError("user_id と notebook_id は必須です")
            scope_clause = "(user_id = %s AND notebook_id = %s)"
            if include_global:
                scope_clause = f"({scope_clause} OR is_global = TRUE)"
            where.append(scope_clause)
            params.extend([scope_user, scope_notebook])

            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            cursor.execute(f"SELECT COUNT(*) FROM documents {where_sql};", params)
            count = cursor.fetchone()[0]

            self.logger.info(f"データベース内のドキュメント数: {count}")
            return count

        except psycopg2.errors.UndefinedTable:
            # テーブルが存在しない場合は0を返す
            self._safe_rollback()  # エラー状態をリセット
            self.logger.info("documentsテーブルが存在しないため、ドキュメント数は0です")
            return 0
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
        ドキュメント一覧を返します（ファイル単位で集約）。
        """
        try:
            self._ensure_connection()

            cursor = self.connection.cursor()
            params: List[Any] = []
            clauses: List[str] = []
            if tenant:
                clauses.append("tenant = %s")
                params.append(tenant)
            if notebook:
                clauses.append("notebook = %s")
                params.append(notebook)
            scope_user = (user_id or "").strip()
            scope_notebook = (notebook_id or "").strip()
            if not scope_user or not scope_notebook:
                raise ValueError("user_id と notebook_id は必須です")
            scope_clause = "(user_id = %s AND notebook_id = %s)"
            if include_global:
                scope_clause = f"({scope_clause} OR is_global = TRUE)"
            clauses.append(scope_clause)
            params.extend([scope_user, scope_notebook])

            where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

            cursor.execute(
                f"""
                SELECT
                    tenant,
                    notebook,
                    MAX(user_id) AS user_id,
                    MAX(notebook_id) AS notebook_id,
                    BOOL_OR(is_global) AS is_global,
                    doc_base_id,
                    COALESCE(NULLIF(MAX(title), ''), doc_base_id) AS title,
                    COUNT(*) AS chunks,
                    MAX(EXTRACT(EPOCH FROM created_at))::BIGINT AS last_updated,
                    MAX(COALESCE(metadata->>'source_file_path', '')) AS source_file_path
                FROM documents
                {where_clause}
                GROUP BY tenant, notebook, doc_base_id
                ORDER BY title;
                """,
                params,
            )

            results = []
            for row in cursor.fetchall():
                row_tenant, row_notebook, row_user_id, row_nb_id, row_is_global, doc_base_id, title, chunks, last_updated, source_file_path = row
                results.append(
                    {
                        "tenant": row_tenant,
                        "notebook": row_notebook,
                        "user_id": row_user_id,
                        "notebook_id": row_nb_id,
                        "is_global": bool(row_is_global),
                        "doc_id": doc_base_id,
                        "title": title,
                        "chunks": int(chunks),
                        "last_updated": int(last_updated) if last_updated is not None else None,
                        "source_file_path": source_file_path,
                    }
                )

            return results
        except Exception as e:
            self.logger.error(f"ドキュメント一覧の取得中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

    def list_notebook_summaries(
        self,
        *,
        user_id: str,
        tenant: Optional[str] = None,
        include_global: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        ノートブックごとの集計情報を返します。

        Args:
            user_id: 対象ユーザー
            tenant: フィルタするテナント（省略可）
            include_global: グローバル資料を含めるかどうか
        """
        try:
            self._ensure_connection()

            scope_user = (user_id or "").strip()
            if not scope_user:
                raise ValueError("user_id は必須です")

            cursor = self.connection.cursor()
            clauses: List[str] = ["notebook_id IS NOT NULL", "notebook_id <> ''"]
            params: List[Any] = []

            if tenant:
                clauses.append("tenant = %s")
                params.append(tenant)

            scope_clause = "user_id = %s"
            params.append(scope_user)
            if include_global:
                scope_clause = f"({scope_clause} OR is_global = TRUE)"
            clauses.append(scope_clause)

            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

            cursor.execute(
                f"""
                SELECT
                    notebook_id,
                    COALESCE(NULLIF(MAX(title), ''), notebook_id) AS title,
                    COUNT(DISTINCT doc_base_id) AS sources,
                    MAX(EXTRACT(EPOCH FROM created_at))::BIGINT AS updated_ts
                FROM documents
                {where_sql}
                GROUP BY notebook_id
                ORDER BY updated_ts DESC NULLS LAST, notebook_id;
                """,
                params,
            )

            summaries: List[Dict[str, Any]] = []
            for row in cursor.fetchall():
                notebook_id, title, sources, updated_ts = row
                summaries.append(
                    {
                        "notebook_id": notebook_id,
                        "title": title,
                        "sources": int(sources),
                        "updated_at": float(updated_ts) if updated_ts is not None else None,
                    }
                )

            return summaries
        except Exception as e:
            self.logger.error(f"ノートブック集計の取得中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

    def delete_document_by_base_id(self, tenant: str, notebook: str, doc_base_id: str) -> int:
        """
        テナント＋doc_base_id でドキュメントを削除します。
        """
        try:
            self._ensure_connection()

            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM documents WHERE tenant = %s AND notebook = %s AND doc_base_id = %s;",
                (tenant, notebook, doc_base_id),
            )
            deleted_rows = cursor.rowcount
            self.connection.commit()
            self.logger.info(f"tenant={tenant} notebook={notebook} doc={doc_base_id} を削除しました（{deleted_rows}件）")
            return deleted_rows
        except Exception as e:
            self._safe_rollback()
            self.logger.error(f"ドキュメント削除中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

    def delete_documents_by_tenant(self, tenant: str, notebook: Optional[str] = None) -> int:
        """
        指定テナントのドキュメントを全削除します。
        """
        try:
            self._ensure_connection()

            cursor = self.connection.cursor()
            if notebook:
                cursor.execute("DELETE FROM documents WHERE tenant = %s AND notebook = %s;", (tenant, notebook))
            else:
                cursor.execute("DELETE FROM documents WHERE tenant = %s;", (tenant,))
            deleted_rows = cursor.rowcount
            self.connection.commit()
            if notebook:
                self.logger.info(f"tenant={tenant} notebook={notebook} のドキュメントを削除しました（{deleted_rows}件）")
            else:
                self.logger.info(f"tenant={tenant} のドキュメントを削除しました（{deleted_rows}件）")
            return deleted_rows
        except Exception as e:
            self._safe_rollback()
            self.logger.error(f"テナント削除中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

    def delete_documents_by_scope(
        self,
        *,
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        user_id: str,
        notebook_id: str,
    ) -> int:
        """
        指定ユーザーのノートブックに紐づくドキュメントを削除します。
        """
        try:
            self._ensure_connection()

            scope_user = (user_id or "").strip()
            scope_notebook_id = (notebook_id or "").strip()
            if not scope_user or not scope_notebook_id:
                raise ValueError("user_id と notebook_id は必須です")

            cursor = self.connection.cursor()

            clauses: List[str] = ["user_id = %s", "notebook_id = %s"]
            params: List[Any] = [scope_user, scope_notebook_id]

            if tenant:
                clauses.append("tenant = %s")
                params.append(tenant)
            if notebook:
                clauses.append("notebook = %s")
                params.append(notebook)

            where_clause = " AND ".join(clauses)
            cursor.execute(f"DELETE FROM documents WHERE {where_clause};", params)
            deleted_rows = cursor.rowcount
            self.connection.commit()
            self.logger.info(
                "ユーザー %s のノートブック %s を削除しました（%d 件）",
                scope_user,
                scope_notebook_id,
                deleted_rows,
            )
            return deleted_rows
        except Exception as e:
            self._safe_rollback()
            self.logger.error(f"ノートブック削除中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

    def migrate_default_scope_to_library(self) -> int:
        """
        旧来の default/default スコープに残っている行を library グローバル棚へ移動する。
        Returns:
            移動した行数
        """
        try:
            self._ensure_connection()

            cursor = self.connection.cursor()
            cursor.execute(
                """
                UPDATE documents
                   SET notebook_id = 'library',
                       is_global = TRUE
                 WHERE user_id = 'default'
                   AND notebook_id = 'default';
                """
            )
            moved = cursor.rowcount
            if moved:
                self.connection.commit()
                self.logger.info(f"defaultスコープからlibraryへ {moved} 件のドキュメントを移動しました")
            else:
                self._safe_rollback()
            return moved
        except Exception as e:
            self._safe_rollback()
            self.logger.error(f"defaultスコープからの移行に失敗しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

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
        ドキュメントの代表的なメタデータを1件返します。
        """
        try:
            self._ensure_connection()

            cursor = self.connection.cursor()
            scope_user = (user_id or "").strip()
            scope_notebook = (notebook_id or "").strip()
            if not scope_user or not scope_notebook:
                raise ValueError("user_id と notebook_id は必須です")

            base_params = [tenant, notebook, doc_base_id, scope_user, scope_notebook]
            where_clause = "tenant = %s AND notebook = %s AND doc_base_id = %s AND user_id = %s AND notebook_id = %s"
            cursor.execute(
                f"""
                SELECT title, metadata
                  FROM documents
                 WHERE {where_clause}
                 ORDER BY created_at DESC
                 LIMIT 1;
                """,
                base_params,
            )
            row = cursor.fetchone()
            if not row:
                return None
            title, metadata_json = row
            metadata = metadata_json if isinstance(metadata_json, dict) else (json.loads(metadata_json) if metadata_json else {})

            # ページ数などの統計を取得
            cursor.execute(
                f"""
                SELECT
                    COUNT(DISTINCT (metadata->>'page')) FILTER (WHERE metadata ? 'page') AS page_count,
                    MAX(metadata->>'source_file_path') AS source_file_path,
                    MAX(metadata->>'processed_file_path') AS processed_file_path,
                    MAX(metadata->>'source_uri') AS source_uri,
                    MAX(metadata->>'txt_uri') AS txt_uri
                  FROM documents
                 WHERE {where_clause};
                """,
                base_params,
            )
            stats_row = cursor.fetchone()
            page_count = stats_row[0] if stats_row else None
            source_file_path = stats_row[1] if stats_row else None
            processed_file_path = stats_row[2] if stats_row else None
            source_uri = stats_row[3] if stats_row else None
            txt_uri = stats_row[4] if stats_row else None

            metadata.setdefault("source_file_path", source_file_path)
            metadata.setdefault("processed_file_path", processed_file_path)
            metadata.setdefault("source_uri", source_uri)
            metadata.setdefault("txt_uri", txt_uri)
            metadata["title"] = title or metadata.get("file_name") or doc_base_id
            metadata["doc_id"] = doc_base_id
            metadata["tenant"] = tenant
            metadata["notebook"] = notebook
            metadata["page_count"] = page_count
            return metadata
        except Exception as e:
            self.logger.error(f"ドキュメントメタデータ取得中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()

    def get_adjacent_chunks(self, file_path: str, chunk_index: int, context_size: int = 1) -> List[Dict[str, Any]]:
        """
        指定されたチャンクの前後のチャンクを取得します。

        Args:
            file_path: ファイルパス
            chunk_index: チャンクインデックス
            context_size: 前後に取得するチャンク数（デフォルト: 1）

        Returns:
            前後のチャンクのリスト

        Raises:
            Exception: 取得に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # 前後のチャンクを取得
            min_index = max(0, chunk_index - context_size)
            max_index = chunk_index + context_size

            cursor.execute(
                """
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 AS similarity
                FROM
                    documents
                WHERE
                    file_path = %s
                    AND chunk_index >= %s
                    AND chunk_index <= %s
                    AND chunk_index != %s
                ORDER BY
                    chunk_index
                """,
                (file_path, min_index, max_index, chunk_index),
            )

            # 結果の取得
            results = []
            for row in cursor.fetchall():
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # メタデータをJSONからデコード
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 既に辞書型の場合はそのまま使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "is_context": True,  # コンテキストチャンクであることを示すフラグ
                    }
                )

            self.logger.info(
                f"ファイル '{file_path}' のチャンク {chunk_index} の前後 {len(results)} 件のチャンクを取得しました"
            )
            return results

        except Exception as e:
            self.logger.error(f"前後のチャンク取得中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def get_document_by_file_path(self, file_path: str) -> List[Dict[str, Any]]:
        """
        指定されたファイルパスに基づいてドキュメント全体を取得します。

        Args:
            file_path: ファイルパス

        Returns:
            ドキュメント全体のチャンクのリスト

        Raises:
            Exception: 取得に失敗した場合
        """
        try:
            # 接続がない場合は接続
            self._ensure_connection()

            # カーソルの作成
            cursor = self.connection.cursor()

            # ファイルパスに基づいてドキュメントを取得
            cursor.execute(
                """
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 AS similarity
                FROM
                    documents
                WHERE
                    file_path = %s
                ORDER BY
                    chunk_index
                """,
                (file_path,),
            )

            # 結果の取得
            results = []
            for row in cursor.fetchall():
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # メタデータをJSONからデコード
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 既に辞書型の場合はそのまま使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "is_full_document": True,  # 全文ドキュメントであることを示すフラグ
                    }
                )

            self.logger.info(f"ファイル '{file_path}' の全文 {len(results)} チャンクを取得しました")
            return results

        except Exception as e:
            self.logger.error(f"ドキュメント全文の取得中にエラーが発生しました: {str(e)}")
            raise

        finally:
            # カーソルを閉じる
            if "cursor" in locals() and cursor:
                cursor.close()

    def get_chunks_for_documents(
        self,
        *,
        doc_ids: Sequence[str],
        tenant: Optional[str] = None,
        notebook: Optional[str] = None,
        user_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        include_global: bool = False,
        max_chunks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        指定した doc_base_id 群に紐づくチャンクをまとめて取得します。
        """
        doc_list = [doc_id.strip() for doc_id in doc_ids if doc_id and doc_id.strip()]
        if not doc_list:
            return []
        try:
            self._ensure_connection()

            scope_user = (user_id or "").strip()
            scope_notebook_id = (notebook_id or "").strip()
            if not scope_user or not scope_notebook_id:
                raise ValueError("user_id と notebook_id は必須です")

            cursor = self.connection.cursor()
            placeholders = ", ".join(["%s"] * len(doc_list))
            where_clauses = [f"doc_base_id IN ({placeholders})"]
            params: List[Any] = list(doc_list)

            if tenant:
                where_clauses.append("tenant = %s")
                params.append(tenant)
            if notebook:
                where_clauses.append("notebook = %s")
                params.append(notebook)

            scope_clause = "(user_id = %s AND notebook_id = %s)"
            if include_global:
                scope_clause = f"({scope_clause} OR is_global = TRUE)"
            where_clauses.append(scope_clause)
            params.extend([scope_user, scope_notebook_id])

            where_sql = " AND ".join(where_clauses)
            query = f"""
                SELECT
                    tenant,
                    notebook,
                    doc_base_id,
                    title,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    user_id,
                    notebook_id,
                    is_global
                FROM documents
                WHERE {where_sql}
                ORDER BY doc_base_id, chunk_index
            """
            if max_chunks:
                query += " LIMIT %s"
                params.append(int(max_chunks))

            cursor.execute(query, params)
            rows = cursor.fetchall() or []
            results: List[Dict[str, Any]] = []
            for row in rows:
                (
                    row_tenant,
                    row_notebook,
                    doc_base_id,
                    title,
                    content,
                    file_path,
                    chunk_index,
                    metadata_json,
                    row_user_id,
                    row_nb_id,
                    row_is_global,
                ) = row

                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "tenant": row_tenant,
                        "notebook": row_notebook,
                        "doc_base_id": doc_base_id,
                        "title": title,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "user_id": row_user_id,
                        "notebook_id": row_nb_id,
                        "is_global": bool(row_is_global),
                    }
                )

            self.logger.info("doc_base_id=%s のチャンクを %d 件取得しました", ",".join(doc_list[:5]), len(results))
            return results
        except Exception as e:
            self.logger.error(f"doc_base_id 指定のチャンク取得中にエラーが発生しました: {str(e)}")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()
