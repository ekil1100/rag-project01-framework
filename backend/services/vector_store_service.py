import os
from datetime import datetime
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
import chromadb
from chromadb.config import Settings
from utils.config import (
    VectorDBProvider,
    MILVUS_CONFIG,
    CHROMA_CONFIG,
)  # Updated import
import re  # Import re module

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息
    """
    def __init__(self, provider: str, index_mode: str):
        """
        初始化向量数据库配置

        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
        """
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.chroma_persist_directory = CHROMA_CONFIG["persist_directory"]

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Milvus索引类型

        参数:
            index_mode: 索引模式

        返回:
            对应的Milvus索引类型
        """
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")

    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取Milvus索引参数

        参数:
            index_mode: 索引模式

        返回:
            对应的Milvus索引参数字典
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})

    def _get_chroma_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取Chroma索引类型

        参数:
            index_mode: 索引模式

        返回:
            对应的Chroma索引类型
        """
        return CHROMA_CONFIG["index_types"].get(index_mode, "Standard")

    def _get_chroma_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取Chroma索引参数

        参数:
            index_mode: 索引模式

        返回:
            对应的Chroma索引参数字典
        """
        return CHROMA_CONFIG["index_params"].get(index_mode, {})


class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能
    """
    def __init__(self):
        """
        初始化向量存储服务
        """
        self.initialized_dbs = {}
        # 确保存储目录存在
        os.makedirs("03-vector-store", exist_ok=True)

    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Milvus索引类型

        参数:
            config: 向量数据库配置对象

        返回:
            Milvus索引类型
        """
        return config._get_milvus_index_type(config.index_mode)

    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Milvus索引参数

        参数:
            config: 向量数据库配置对象

        返回:
            Milvus索引参数字典
        """
        return config._get_milvus_index_params(config.index_mode)

    def _get_chroma_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取Chroma索引类型

        参数:
            config: 向量数据库配置对象

        返回:
            Chroma索引类型
        """
        return config._get_chroma_index_type(config.index_mode)

    def _get_chroma_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取Chroma索引参数

        参数:
            config: 向量数据库配置对象

        返回:
            Chroma索引参数字典
        """
        return config._get_chroma_index_params(config.index_mode)

    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到向量数据库

        参数:
            embedding_file: 嵌入向量文件路径
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        start_time = datetime.now()

        # 读取embedding文件
        embeddings_data = self._load_embeddings(embedding_file)

        # 根据不同的数据库进行索引
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)
        elif config.provider == VectorDBProvider.CHROMA:
            result = self._index_to_chroma(embeddings_data, config)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A")
        }

    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        加载embedding文件，返回配置信息和embeddings

        参数:
            file_path: 嵌入向量文件路径

        返回:
            包含嵌入向量和元数据的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")

                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")

                # 返回完整的数据，包括顶层配置
                logger.info(f"Found {len(data['embeddings'])} embeddings")
                return data

        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise

    def _index_to_milvus(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将嵌入向量索引到Milvus数据库

        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        try:
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀，移除它
            base_name_raw = filename.replace(".pdf", "") if filename else "doc"

            # --- Sanitize base_name for Milvus --- Start
            # Replace invalid characters (not letter, number, or underscore) with underscore
            sanitized_base_name = re.sub(r"[^a-zA-Z0-9_]", "_", base_name_raw)
            # Replace multiple consecutive underscores with a single underscore
            sanitized_base_name = re.sub(r"_+", "_", sanitized_base_name)
            # Remove leading and trailing underscores
            sanitized_base_name = sanitized_base_name.strip("_")

            # If sanitized name is empty, use default 'doc'
            if not sanitized_base_name:
                sanitized_base_name = "doc"

            # Ensure the collection name starts with a letter or underscore
            # (Milvus requirement)
            if not sanitized_base_name[0].isalpha() and sanitized_base_name[0] != "_":
                sanitized_base_name = f"m_{sanitized_base_name}"  # Prepend 'm_' if it doesn't start correctly
            # --- Sanitize base_name for Milvus --- End

            # Get embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # Use sanitized name
            collection_name = f"{sanitized_base_name}_{embedding_provider}_{timestamp}"

            # 连接到Milvus
            connections.connect(alias="default", uri=config.milvus_uri)

            # 从顶层配置获取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")

            logger.info(f"Creating collection with dimension: {vector_dim}")

            # Define a specific name for the vector index
            vector_index_name = "vector_idx"

            # Define fields
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": 5000},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                {"name": "embedding_provider", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_model", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_timestamp", "dtype": "VARCHAR", "max_length": 50},
                {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": vector_dim},
            ]

            # Prepare data
            entities = []
            for emb in embeddings_data["embeddings"]:
                entity = {
                    "content": str(emb["metadata"].get("content", "")),
                    "document_name": embeddings_data.get(
                        "filename", ""
                    ),  # 使用 filename 而不是 document_name
                    "chunk_id": int(emb["metadata"].get("chunk_id", 0)),
                    "total_chunks": int(emb["metadata"].get("total_chunks", 0)),
                    "word_count": int(emb["metadata"].get("word_count", 0)),
                    "page_number": str(emb["metadata"].get("page_number", 0)),
                    "page_range": str(emb["metadata"].get("page_range", "")),
                    "embedding_provider": embeddings_data.get(
                        "embedding_provider", ""
                    ),  # 从顶层配置获取
                    "embedding_model": embeddings_data.get(
                        "embedding_model", ""
                    ),  # 从顶层配置获取
                    "embedding_timestamp": str(
                        emb["metadata"].get("embedding_timestamp", "")
                    ),
                    "vector": [float(x) for x in emb.get("embedding", [])],
                }
                entities.append(entity)

            logger.info(f"Target Milvus collection: {collection_name}")

            # Create collection schema
            field_schemas = []
            for field in fields:
                extra_params = {}
                if field.get('max_length') is not None:
                    extra_params['max_length'] = field['max_length']
                if field.get('dim') is not None:
                    extra_params["dim"] = field["dim"]
                field_schema = FieldSchema(
                    name=field["name"],
                    dtype=getattr(DataType, field["dtype"]),
                    is_primary=field.get("is_primary", False),
                    auto_id=field.get("auto_id", False),
                    **extra_params,
                )
                field_schemas.append(field_schema)

            schema = CollectionSchema(fields=field_schemas, description=f"Collection for {collection_name}")

            # Get or create collection
            if not utility.has_collection(collection_name):
                collection = Collection(name=collection_name, schema=schema)
                logger.info(f"Collection {collection_name} created.")
            else:
                collection = Collection(name=collection_name)
                logger.info(f"Using existing collection {collection_name}.")

            # Insert data
            logger.info(f"Inserting {len(entities)} vectors into {collection_name}")
            insert_result = collection.insert(entities)
            collection.flush()
            logger.info(f"Flushed {len(insert_result.primary_keys)} inserted vectors.")

            # Create index only if our specific index doesn't exist
            # Check for the specific index name
            if not collection.has_index(index_name=vector_index_name):
                logger.info(
                    f"Creating index '{vector_index_name}' for collection {collection_name}"
                )
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": self._get_milvus_index_type(config),
                    "params": self._get_milvus_index_params(config),
                }
                # Specify the index name during creation
                collection.create_index(
                    field_name="vector",
                    index_params=index_params,
                    index_name=vector_index_name,
                )
                logger.info(
                    f"Index '{vector_index_name}' created with params: {index_params}"
                )
            else:
                logger.info(
                    f"Index '{vector_index_name}' already exists for collection {collection_name}"
                )

            # Load collection into memory
            collection.load()
            logger.info(f"Collection {collection_name} loaded.")

            return {
                "index_size": len(insert_result.primary_keys),
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            raise

        finally:
            # Ensure disconnection happens even if errors occur
            try:
                connections.disconnect("default")
                logger.info("Disconnected from Milvus.")
            except Exception as disconnect_e:
                logger.error(f"Error disconnecting from Milvus: {disconnect_e}")

    def _index_to_chroma(
        self, embeddings_data: Dict[str, Any], config: VectorDBConfig
    ) -> Dict[str, Any]:
        """
        将嵌入向量索引到Chroma数据库

        参数:
            embeddings_data: 嵌入向量数据
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        try:
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀，移除它
            base_name_raw = filename.replace(".pdf", "") if filename else "doc"

            # --- Sanitize base_name for Chroma --- Start
            # 替换空格和特殊字符为下划线
            sanitized_base_name = re.sub(r"[^a-zA-Z0-9_-]", "_", base_name_raw)
            # 移除连续的下划线
            sanitized_base_name = re.sub(r"_+", "_", sanitized_base_name)
            # 移除开头和结尾的下划线
            sanitized_base_name = sanitized_base_name.strip("_")

            # 如果清理后为空，使用默认名称
            if not sanitized_base_name:
                sanitized_base_name = "doc"
            # --- Sanitize base_name for Chroma --- End

            # 获取embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # 构造初步的 collection 名称
            base_collection_name = (
                f"{sanitized_base_name}_{embedding_provider}_{timestamp}"
            )

            # --- Truncate collection_name if necessary --- Start
            max_len = 63
            if len(base_collection_name) > max_len:
                # 保留时间戳和部分 provider，截断 base_name
                suffix = f"_{embedding_provider}_{timestamp}"
                allowed_base_len = (
                    max_len - len(suffix) - 1
                )  # -1 for potential leading _ if base is empty
                if allowed_base_len <= 0:
                    # Should not happen with timestamp, but as fallback use truncated timestamp
                    collection_name = timestamp[:max_len]
                else:
                    truncated_base = sanitized_base_name[:allowed_base_len]
                    # Ensure it doesn't end with _ after truncation
                    collection_name = f"{truncated_base.strip('_')}{suffix}"
            else:
                collection_name = base_collection_name

            # Final check for start/end character (should be handled by sanitization/truncation)
            if not collection_name[0].isalnum():
                collection_name = (
                    f"c{collection_name[1:]}"  # Prepend 'c' if starts with non-alnum
                )
            if not collection_name[-1].isalnum():
                collection_name = (
                    f"{collection_name[:-1]}c"  # Append 'c' if ends with non-alnum
                )

            # Ensure length is at least 3
            if len(collection_name) < 3:
                collection_name = (
                    f"{collection_name}{timestamp[:3-len(collection_name)]}"
                )
            # --- Truncate collection_name if necessary --- End

            logger.info(
                f"Sanitized and finalized Chroma collection name: {collection_name}"
            )

            # 确保持久化目录存在
            os.makedirs(config.chroma_persist_directory, exist_ok=True)

            # 从顶层配置获取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")

            logger.info(f"Creating Chroma collection with dimension: {vector_dim}")

            # 创建Chroma客户端
            chroma_client = chromadb.PersistentClient(
                path=config.chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # 创建collection
            # Collection metadata can contain complex types like dicts
            collection_metadata = {
                "description": f"Collection for {collection_name}",
                "embedding_provider": embedding_provider,
                "embedding_model": embeddings_data.get("embedding_model", ""),
                "vector_dimension": vector_dim,
                "index_type": self._get_chroma_index_type(config),
                # Ensure index_params is serializable if it's complex
                "index_params": (
                    json.dumps(self._get_chroma_index_params(config))
                    if self._get_chroma_index_params(config)
                    else None
                ),
            }
            # Filter out None values from collection metadata
            collection_metadata = {
                k: v for k, v in collection_metadata.items() if v is not None
            }

            collection = chroma_client.create_collection(
                name=collection_name,
                metadata=collection_metadata,
            )

            # 准备数据
            ids = []
            documents = []
            metadatas = []
            embeddings = []

            # Define allowed types for Chroma metadata values
            ALLOWED_METADATA_TYPES = (str, int, float, bool)

            for idx, emb in enumerate(embeddings_data["embeddings"]):
                # 唯一ID
                ids.append(f"{collection_name}_{idx}")

                # 文档内容 - ensure it's a string
                documents.append(str(emb["metadata"].get("content", "")))

                # 元数据 - ensure all values are of allowed types
                raw_metadata = {
                    "document_name": embeddings_data.get("filename", ""),
                    "chunk_id": emb["metadata"].get("chunk_id"),
                    "total_chunks": emb["metadata"].get("total_chunks"),
                    "word_count": emb["metadata"].get("word_count"),
                    "page_number": emb["metadata"].get("page_number"),
                    "page_range": emb["metadata"].get("page_range"),
                    "embedding_provider": embeddings_data.get("embedding_provider"),
                    "embedding_model": embeddings_data.get("embedding_model"),
                    "embedding_timestamp": emb["metadata"].get("embedding_timestamp"),
                }

                # Clean and convert metadata values
                cleaned_metadata = {}
                for key, value in raw_metadata.items():
                    if value is None:
                        continue  # Skip None values
                    elif isinstance(value, ALLOWED_METADATA_TYPES):
                        cleaned_metadata[key] = value
                    elif isinstance(value, (dict, list)):
                        # Convert dicts/lists to JSON strings if necessary
                        try:
                            cleaned_metadata[key] = json.dumps(value)
                        except TypeError:
                            # Fallback to string representation if JSON fails
                            cleaned_metadata[key] = str(value)
                    else:
                        # Attempt to convert other types to string
                        cleaned_metadata[key] = str(value)

                metadatas.append(cleaned_metadata)

                # 向量 - ensure they are lists of floats
                embedding = [
                    float(x)
                    for x in emb.get("embedding", [])
                    if isinstance(x, (int, float))
                ]
                # Handle potential empty embedding after filtering
                if not embedding and emb.get("embedding") is not None:
                    logger.warning(
                        f"Could not parse embedding for ID {ids[-1]}. Original: {emb.get('embedding')}"
                    )
                    # Decide how to handle: skip, add zeros, etc.
                    # Skipping for now:
                    # ids.pop()
                    # documents.pop()
                    # metadatas.pop()
                    # continue
                    # Or adding zeros (ensure dimension matches!)
                    embedding = [0.0] * vector_dim

                embeddings.append(embedding)

            logger.info(f"Adding {len(ids)} documents to Chroma collection")

            # 批量添加文档
            if ids:  # Only add if there are documents to add
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                logger.info(f"Successfully indexed {len(ids)} documents to Chroma")
            else:
                logger.warning("No valid documents found to index to Chroma.")

            return {"index_size": len(ids), "collection_name": collection_name}

        except Exception as e:
            logger.error(
                f"Error indexing to Chroma: {str(e)}", exc_info=True
            )  # Add traceback
            raise

    def list_collections(self, provider: str) -> List[str]:
        """
        列出指定提供商的所有集合

        参数:
            provider: 向量数据库提供商

        返回:
            集合名称列表
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collections = utility.list_collections()
                return collections
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                os.makedirs(CHROMA_CONFIG["persist_directory"], exist_ok=True)
                chroma_client = chromadb.PersistentClient(
                    path=CHROMA_CONFIG["persist_directory"],
                    settings=Settings(anonymized_telemetry=False),
                )
                collections = chroma_client.list_collections()
                return [col.name for col in collections]
            except Exception as e:
                logger.error(f"Error listing Chroma collections: {str(e)}")
                return []
        return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定的集合

        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称

        返回:
            是否删除成功
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                utility.drop_collection(collection_name)
                return True
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                chroma_client = chromadb.PersistentClient(
                    path=CHROMA_CONFIG["persist_directory"],
                    settings=Settings(anonymized_telemetry=False),
                )
                chroma_client.delete_collection(collection_name)
                return True
            except Exception as e:
                logger.error(f"Error deleting Chroma collection: {str(e)}")
                return False
        return False

    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """
        获取指定集合的信息

        参数:
            provider: 向量数据库提供商
            collection_name: 集合名称

        返回:
            集合信息字典
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collection = Collection(collection_name)
                return {
                    "name": collection_name,
                    "num_entities": collection.num_entities,
                    "schema": collection.schema.to_dict()
                }
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA:
            try:
                chroma_client = chromadb.PersistentClient(
                    path=CHROMA_CONFIG["persist_directory"],
                    settings=Settings(anonymized_telemetry=False),
                )
                collection = chroma_client.get_collection(collection_name)
                count = collection.count()
                return {
                    "name": collection_name,
                    "num_entities": count,
                    "schema": {"metadata": collection.metadata},
                }
            except Exception as e:
                logger.error(f"Error getting Chroma collection info: {str(e)}")
                return {}
        return {}
