"""
ドキュメント処理モジュール

マークダウン、テキスト、パワーポイント、PDFなどのファイルの読み込みと解析、チャンク分割を行います。
"""

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import hashlib
import time

import markitdown
from pypdf import PdfReader


class DocumentProcessor:
    """
    ドキュメント処理クラス

    マークダウン、テキスト、パワーポイント、PDFなどのファイルの読み込みと解析、チャンク分割を行います。

    Attributes:
        logger: ロガー
    """

    # サポートするファイル拡張子
    SUPPORTED_EXTENSIONS = {
        "text": [".txt", ".md", ".markdown"],
        "office": [".ppt", ".pptx", ".doc", ".docx"],
        "pdf": [".pdf"],
    }

    def __init__(self):
        """
        DocumentProcessorのコンストラクタ
        """
        # ロガーの設定
        self.logger = logging.getLogger("document_processor")
        self.logger.setLevel(logging.INFO)

    def read_file(self, file_path: str) -> str:
        """
        ファイルを読み込みます。

        Args:
            file_path: ファイルのパス

        Returns:
            ファイルの内容

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            IOError: ファイルの読み込みに失敗した場合
        """
        try:
            # ファイル拡張子を取得
            ext = Path(file_path).suffix.lower()

            # テキストファイル（マークダウン含む）の場合
            if ext in self.SUPPORTED_EXTENSIONS["text"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # NUL文字を削除
                    content = content.replace("\x00", "")
                self.logger.info(f"テキストファイル '{file_path}' を読み込みました")
                return content

            # パワーポイント、Word、PDFの場合はmarkitdownを使用して変換
            elif ext in self.SUPPORTED_EXTENSIONS["office"] or ext in self.SUPPORTED_EXTENSIONS["pdf"]:
                return self.convert_to_markdown(file_path)

            # サポートしていない拡張子の場合
            else:
                self.logger.warning(f"サポートしていないファイル形式です: {file_path}")
                return ""

        except FileNotFoundError:
            self.logger.error(f"ファイル '{file_path}' が見つかりません")
            raise
        except IOError as e:
            self.logger.error(f"ファイル '{file_path}' の読み込みに失敗しました: {str(e)}")
            raise

    def convert_to_markdown(self, file_path: str) -> str:
        """
        パワーポイント、Word、PDFなどのファイルをマークダウンに変換します。

        Args:
            file_path: ファイルのパス

        Returns:
            マークダウンに変換された内容

        Raises:
            Exception: 変換に失敗した場合
        """
        try:
            # ファイルURIを作成
            file_uri = f"file://{os.path.abspath(file_path)}"

            # markitdownを使用して変換
            markdown_content = markitdown.MarkItDown().convert_uri(file_uri).markdown
            # NUL文字を削除
            markdown_content = markdown_content.replace("\x00", "")

            self.logger.info(f"ファイル '{file_path}' をマークダウンに変換しました")
            return markdown_content
        except Exception as e:
            self.logger.error(f"ファイル '{file_path}' のマークダウン変換に失敗しました: {str(e)}")
            raise

    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        テキストをチャンクに分割します。

        Args:
            text: 分割するテキスト
            chunk_size: チャンクサイズ（文字数）
            overlap: チャンク間のオーバーラップ（文字数）

        Returns:
            チャンクのリスト
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # 文の途中で切らないように調整
            if end < text_length:
                # 次の改行または句点を探す
                next_newline = text.find("\n", end)
                next_period = text.find("。", end)

                if next_newline != -1 and (next_period == -1 or next_newline < next_period):
                    end = next_newline + 1  # 改行を含める
                elif next_period != -1:
                    end = next_period + 1  # 句点を含める

            chunks.append(text[start:end])
            start = end - overlap if end - overlap > start else end

            # 終了条件
            if start >= text_length:
                break

        self.logger.info(f"テキストを {len(chunks)} チャンクに分割しました")
        return chunks

    def calculate_file_hash(self, file_path: str) -> str:
        """
        ファイルのハッシュ値を計算します。

        Args:
            file_path: ファイルのパス

        Returns:
            ファイルのSHA-256ハッシュ値
        """
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            self.logger.error(f"ファイル '{file_path}' のハッシュ計算に失敗しました: {str(e)}")
            # エラーが発生した場合は、タイムスタンプをハッシュとして使用
            return f"timestamp-{int(time.time())}"

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        ファイルのメタデータを取得します。

        Args:
            file_path: ファイルのパス

        Returns:
            ファイルのメタデータ（ハッシュ値、最終更新日時など）
        """
        file_stat = os.stat(file_path)
        return {
            "hash": self.calculate_file_hash(file_path),
            "mtime": file_stat.st_mtime,
            "size": file_stat.st_size,
            "path": file_path,
        }

    def load_file_registry(self, processed_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        処理済みファイルのレジストリを読み込みます。

        Args:
            processed_dir: 処理済みファイルを保存するディレクトリのパス

        Returns:
            処理済みファイルのレジストリ（ファイルパスをキーとするメタデータの辞書）
        """
        registry_path = Path(processed_dir) / "file_registry.json"
        if not registry_path.exists():
            return {}

        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"ファイルレジストリの読み込みに失敗しました: {str(e)}")
            return {}

    def save_file_registry(self, processed_dir: str, registry: Dict[str, Dict[str, Any]]) -> None:
        """
        処理済みファイルのレジストリを保存します。

        Args:
            processed_dir: 処理済みファイルを保存するディレクトリのパス
            registry: 処理済みファイルのレジストリ
        """
        registry_path = Path(processed_dir) / "file_registry.json"
        try:
            # 処理済みディレクトリが存在しない場合は作成
            os.makedirs(Path(processed_dir), exist_ok=True)

            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            self.logger.info(f"ファイルレジストリを保存しました: {registry_path}")
        except Exception as e:
            self.logger.error(f"ファイルレジストリの保存に失敗しました: {str(e)}")

    def _determine_suffix(self, file_path: Path) -> Tuple[str, str]:
        """
        ファイル名とディレクトリサフィックスを計算するヘルパー
        """
        try:
            relative_path = file_path.relative_to(Path(file_path.parts[0]) / Path(file_path.parts[1]))
            parent_dirs = relative_path.parent.parts
            dir_suffix = "_".join(parent_dirs) if parent_dirs else ""
        except ValueError:
            dir_suffix = ""
        processed_file_name = f"{file_path.stem}{('_' + dir_suffix) if dir_suffix else ''}.md"
        return processed_file_name, dir_suffix

    def _ensure_processed_file(self, processed_dir: str, processed_file_name: str, content: str) -> Path:
        """
        処理済みテキストを保存してパスを返す
        """
        processed_file_path = Path(processed_dir) / processed_file_name
        os.makedirs(Path(processed_dir), exist_ok=True)
        with open(processed_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.logger.info(f"処理済みファイルを保存しました: {processed_file_path}")
        return processed_file_path

    def _iter_pdf_pages(self, file_path: str) -> Iterable[Tuple[int, str]]:
        """
        PDF からページ単位でテキストを抽出する
        """
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                self.logger.warning(f"PDFページ抽出に失敗しました (page={index}, file={file_path}): {exc}")
                text = ""
            yield index, text

    def process_file(
        self, file_path: str, processed_dir: str, chunk_size: int = 500, overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        ファイルを処理します。
        """
        try:
            file_path_obj = Path(file_path)
            processed_file_name, dir_suffix = self._determine_suffix(file_path_obj)

            # ファイルの種類に応じて処理
            ext = file_path_obj.suffix.lower()
            if ext in self.SUPPORTED_EXTENSIONS["pdf"]:
                markdown_content = self.read_file(file_path)
                processed_file_path = self._ensure_processed_file(processed_dir, processed_file_name, markdown_content)
                return self._process_pdf_chunks(
                    file_path=file_path,
                    processed_file_path=processed_file_path,
                    dir_suffix=dir_suffix,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

            # その他のファイル
            content = self.read_file(file_path)
            if not content:
                return []

            processed_file_path = self._ensure_processed_file(processed_dir, processed_file_name, content)
            chunks = self.split_into_chunks(content, chunk_size, overlap)

            results: List[Dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                results.append(
                    {
                        "document_id": f"{processed_file_name}_{i}",
                        "content": chunk,
                        "file_path": str(processed_file_path),
                        "original_file_path": file_path,
                        "chunk_index": i,
                        "metadata": {
                            "file_name": file_path_obj.name,
                            "directory": str(file_path_obj.parent),
                            "directory_suffix": dir_suffix,
                            "page": 1,
                            "source_uri": os.path.abspath(file_path),
                            "txt_uri": str(processed_file_path),
                            "offsets": None,
                        },
                    }
                )

            self.logger.info(f"ファイル '{file_path}' を処理しました（{len(results)} チャンク）")
            return results

        except Exception as e:
            self.logger.error(f"ファイル '{file_path}' の処理中にエラーが発生しました: {str(e)}")
            raise

    def _process_pdf_chunks(
        self,
        file_path: str,
        processed_file_path: Path,
        dir_suffix: str,
        chunk_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        """
        PDF をページ単位でチャンク化し、ページ情報をメタデータに含める
        """
        results: List[Dict[str, Any]] = []
        file_path_obj = Path(file_path)
        processed_file_name = processed_file_path.name
        chunk_counter = 0

        for page_no, page_text in self._iter_pdf_pages(file_path):
            if not page_text:
                continue
            page_chunks = self.split_into_chunks(page_text, chunk_size, overlap)
            for chunk in page_chunks:
                results.append(
                    {
                        "document_id": f"{processed_file_name}_{chunk_counter}",
                        "content": chunk,
                        "file_path": str(processed_file_path),
                        "original_file_path": file_path,
                        "chunk_index": chunk_counter,
                        "metadata": {
                            "file_name": file_path_obj.name,
                            "directory": str(file_path_obj.parent),
                            "directory_suffix": dir_suffix,
                            "page": page_no,
                            "source_uri": os.path.abspath(file_path),
                            "txt_uri": str(processed_file_path),
                            "offsets": None,
                        },
                    }
                )
                chunk_counter += 1

        if not results:
            self.logger.warning(f"PDF '{file_path}' からテキストが抽出できませんでした。")

        return results

    def process_directory(
        self, source_dir: str, processed_dir: str, chunk_size: int = 500, overlap: int = 100, incremental: bool = False
    ) -> List[Dict[str, Any]]:
        """
        ディレクトリ内のファイルを処理します。

        Args:
            source_dir: 原稿ファイルが含まれるディレクトリのパス
            processed_dir: 処理済みファイルを保存するディレクトリのパス
            chunk_size: チャンクサイズ（文字数）
            overlap: チャンク間のオーバーラップ（文字数）
            incremental: 差分のみを処理するかどうか

        Returns:
            処理結果のリスト（各要素はチャンク情報を含む辞書）
        """
        results = []
        source_directory = Path(source_dir)

        if not source_directory.exists() or not source_directory.is_dir():
            self.logger.error(f"ディレクトリ '{source_dir}' が見つからないか、ディレクトリではありません")
            raise FileNotFoundError(f"ディレクトリ '{source_dir}' が見つからないか、ディレクトリではありません")

        # サポートするファイル拡張子を全て取得
        all_extensions = []
        for ext_list in self.SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(ext_list)

        # ファイルを検索
        files = []
        for ext in all_extensions:
            files.extend(list(source_directory.glob(f"**/*{ext}")))

        self.logger.info(f"ディレクトリ '{source_dir}' 内に {len(files)} 個のファイルが見つかりました")

        # 差分処理の場合、ファイルレジストリを読み込む
        if incremental:
            file_registry = self.load_file_registry(processed_dir)
            self.logger.info(f"ファイルレジストリから {len(file_registry)} 個のファイル情報を読み込みました")
        else:
            file_registry = {}

        # 処理対象のファイルを特定
        files_to_process = []
        for file_path in files:
            str_path = str(file_path)
            if incremental:
                # ファイルのメタデータを取得
                current_metadata = self.get_file_metadata(str_path)

                # レジストリに存在しない、またはハッシュ値が変更されている場合のみ処理
                if (
                    str_path not in file_registry
                    or file_registry[str_path]["hash"] != current_metadata["hash"]
                    or file_registry[str_path]["mtime"] != current_metadata["mtime"]
                    or file_registry[str_path]["size"] != current_metadata["size"]
                ):
                    files_to_process.append(file_path)
                    # レジストリを更新
                    file_registry[str_path] = current_metadata
            else:
                # 差分処理でない場合は全てのファイルを処理
                files_to_process.append(file_path)
                # レジストリを更新
                file_registry[str_path] = self.get_file_metadata(str_path)

        self.logger.info(f"処理対象のファイル数: {len(files_to_process)} / {len(files)}")

        # 各ファイルを処理
        for file_path in files_to_process:
            try:
                file_results = self.process_file(str(file_path), processed_dir, chunk_size, overlap)
                results.extend(file_results)
            except Exception as e:
                self.logger.error(f"ファイル '{file_path}' の処理中にエラーが発生しました: {str(e)}")
                # エラーが発生しても処理を続行
                continue

        # ファイルレジストリを保存
        self.save_file_registry(processed_dir, file_registry)

        self.logger.info(f"ディレクトリ '{source_dir}' 内のファイルを処理しました（合計 {len(results)} チャンク）")
        return results
