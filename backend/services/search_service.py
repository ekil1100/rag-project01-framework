from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pymilvus import connections, Collection, utility
import chromadb
from chromadb.config import Settings
from services.embedding_service import EmbeddingService
from utils.config import VectorDBProvider, MILVUS_CONFIG, CHROMA_CONFIG
import os
import json

logger = logging.getLogger(__name__)

class SearchService:
    """
    搜索服务类，负责向量数据库的连接和向量搜索功能
    提供集合列表查询、向量相似度搜索和搜索结果保存等功能
    """
    def __init__(self):
        """
        初始化搜索服务
        创建嵌入服务实例，设置Milvus连接URI，初始化搜索结果保存目录
        """
        self.embedding_service = EmbeddingService()
        self.milvus_uri = MILVUS_CONFIG["uri"]
        self.chroma_persist_directory = CHROMA_CONFIG["persist_directory"]
        self.search_results_dir = "04-search-results"
        os.makedirs(self.search_results_dir, exist_ok=True)

    def get_providers(self) -> List[Dict[str, str]]:
        """
        获取支持的向量数据库列表

        Returns:
            List[Dict[str, str]]: 支持的向量数据库提供商列表
        """
        return [
            {"id": VectorDBProvider.MILVUS.value, "name": "Milvus"},
            {"id": VectorDBProvider.CHROMA.value, "name": "Chroma"},
        ]

    def list_collections(self, provider: str = VectorDBProvider.MILVUS.value) -> List[Dict[str, Any]]:
        """
        获取指定向量数据库中的所有集合

        Args:
            provider (str): 向量数据库提供商，默认为Milvus

        Returns:
            List[Dict[str, Any]]: 集合信息列表，包含id、名称和实体数量

        Raises:
            Exception: 连接或查询集合时发生错误
        """
        if provider == VectorDBProvider.MILVUS.value:
            try:
                connections.connect(alias="default", uri=self.milvus_uri)

                collections = []
                collection_names = utility.list_collections()

                for name in collection_names:
                    try:
                        collection = Collection(name)
                        collections.append(
                            {"id": name, "name": name, "count": collection.num_entities}
                        )
                    except Exception as e:
                        logger.error(
                            f"Error getting info for collection {name}: {str(e)}"
                        )

                return collections

            except Exception as e:
                logger.error(f"Error listing collections: {str(e)}")
                raise
            finally:
                connections.disconnect("default")
        elif provider == VectorDBProvider.CHROMA.value:
            try:
                collections = []
                os.makedirs(self.chroma_persist_directory, exist_ok=True)

                chroma_client = chromadb.PersistentClient(
                    path=self.chroma_persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )

                collection_list = chroma_client.list_collections()

                for col in collection_list:
                    try:
                        collection = chroma_client.get_collection(col.name)
                        count = collection.count()
                        collections.append(
                            {"id": col.name, "name": col.name, "count": count}
                        )
                    except Exception as e:
                        logger.error(
                            f"Error getting info for Chroma collection {col.name}: {str(e)}"
                        )

                return collections

            except Exception as e:
                logger.error(f"Error listing Chroma collections: {str(e)}")
                return []

        return []

    def save_search_results(self, query: str, collection_id: str, results: List[Dict[str, Any]]) -> str:
        """
        保存搜索结果到JSON文件

        Args:
            query (str): 搜索查询文本
            collection_id (str): 集合ID
            results (List[Dict[str, Any]]): 搜索结果列表

        Returns:
            str: 保存文件的路径

        Raises:
            Exception: 保存文件时发生错误
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            # 使用集合ID的基础名称（去掉路径相关字符）
            collection_base = os.path.basename(collection_id)
            filename = f"search_{collection_base}_{timestamp}.json"
            filepath = os.path.join(self.search_results_dir, filename)

            search_data = {
                "query": query,
                "collection_id": collection_id,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }

            logger.info(f"Saving search results to: {filepath}")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(search_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved search results to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving search results: {str(e)}")
            raise

    async def search(
        self,
        query: str,
        collection_id: str,
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 20,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        执行向量搜索

        Args:
            query (str): 搜索查询文本
            collection_id (str): 要搜索的集合ID
            top_k (int): 返回的最大结果数量，默认为3
            threshold (float): 相似度阈值，低于此值的结果将被过滤，默认为0.7
            word_count_threshold (int): 文本字数阈值，低于此值的结果将被过滤，默认为20
            save_results (bool): 是否保存搜索结果，默认为False

        Returns:
            Dict[str, Any]: 包含搜索结果的字典，如果保存结果则包含保存路径

        Raises:
            Exception: 搜索过程中发生错误
        """
        # 默认为Milvus
        provider = VectorDBProvider.MILVUS.value

        # 尝试从集合名称判断数据库类型 (修复逻辑)
        try:
            chroma_collections = self.list_collections(VectorDBProvider.CHROMA.value)
            is_chroma_collection = any(
                coll_info["id"] == collection_id for coll_info in chroma_collections
            )
        except Exception as e:
            # If listing Chroma collections fails, log warning and proceed assuming Milvus (or handle differently)
            logger.warning(
                f"Could not list Chroma collections to determine provider for {collection_id}: {e}"
            )
            is_chroma_collection = False

        if is_chroma_collection:
            provider = VectorDBProvider.CHROMA.value
        # else: provider remains Milvus (default)

        logger.info(f"Detected provider for collection {collection_id}: {provider}")

        if provider == VectorDBProvider.MILVUS.value:
            return await self._search_milvus(
                query=query,
                collection_id=collection_id,
                top_k=top_k,
                threshold=threshold,
                word_count_threshold=word_count_threshold,
                save_results=save_results,
            )
        elif provider == VectorDBProvider.CHROMA.value:
            return await self._search_chroma(
                query=query,
                collection_id=collection_id,
                top_k=top_k,
                threshold=threshold,
                word_count_threshold=word_count_threshold,
                save_results=save_results,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _search_milvus(
        self,
        query: str,
        collection_id: str,
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 20,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        在Milvus中执行向量搜索

        Args:
            query (str): 搜索查询文本
            collection_id (str): 要搜索的集合ID
            top_k (int): 返回的最大结果数量
            threshold (float): 相似度阈值
            word_count_threshold (int): 文本字数阈值
            save_results (bool): 是否保存搜索结果

        Returns:
            Dict[str, Any]: 包含搜索结果的字典
        """
        try:
            # 添加参数日志
            logger.info(f"Milvus search parameters:")
            logger.info(f"- Query: {query}")
            logger.info(f"- Collection ID: {collection_id}")
            logger.info(f"- Top K: {top_k}")
            logger.info(f"- Threshold: {threshold}")
            logger.info(f"- Word Count Threshold: {word_count_threshold}")
            logger.info(f"- Save Results: {save_results} (type: {type(save_results)})")

            logger.info(
                f"Starting Milvus search with parameters - Collection: {collection_id}, Query: {query}, Top K: {top_k}"
            )

            # 连接到 Milvus
            logger.info(f"Connecting to Milvus at {self.milvus_uri}")
            connections.connect(
                alias="default",
                uri=self.milvus_uri
            )

            # 获取collection
            logger.info(f"Loading collection: {collection_id}")
            collection = Collection(collection_id)
            collection.load()

            # 记录collection的基本信息
            logger.info(f"Collection info - Entities: {collection.num_entities}")

            # 从collection中读取embedding配置
            logger.info("Querying sample entity for embedding configuration")
            sample_entity = collection.query(
                expr="id >= 0",
                output_fields=["embedding_provider", "embedding_model"],
                limit=1,
            )
            if not sample_entity:
                logger.error(f"Collection {collection_id} is empty")
                raise ValueError(f"Collection {collection_id} is empty")

            logger.info(f"Sample entity configuration: {sample_entity[0]}")

            # 使用collection中存储的配置创建查询向量
            logger.info("Creating query embedding")
            query_embedding = self.embedding_service.create_single_embedding(
                query,
                provider=sample_entity[0]["embedding_provider"],
                model=sample_entity[0]["embedding_model"]
            )
            logger.info(f"Query embedding created with dimension: {len(query_embedding)}")

            # 执行搜索
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            logger.info(f"Executing search with params: {search_params}")
            logger.info(f"Word count threshold filter: word_count >= {word_count_threshold}")

            results = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=f"word_count >= {word_count_threshold}",
                output_fields=[
                    "content",
                    "document_name",
                    "chunk_id",
                    "total_chunks",
                    "word_count",
                    "page_number",
                    "page_range",
                    "embedding_provider",
                    "embedding_model",
                    "embedding_timestamp"
                ]
            )

            # 处理结果
            processed_results = []
            logger.info(f"Raw search results count: {len(results[0])}")

            for hits in results:
                for hit in hits:
                    logger.info(f"Processing hit - Score: {hit.score}, Word Count: {hit.entity.get('word_count')}")
                    if hit.score >= threshold:
                        processed_results.append({
                            "text": hit.entity.content,
                            "score": float(hit.score),
                            "metadata": {
                                "source": hit.entity.document_name,
                                "page": hit.entity.page_number,
                                "chunk": hit.entity.chunk_id,
                                "total_chunks": hit.entity.total_chunks,
                                "page_range": hit.entity.page_range,
                                "embedding_provider": hit.entity.embedding_provider,
                                "embedding_model": hit.entity.embedding_model,
                                "embedding_timestamp": hit.entity.embedding_timestamp
                            }
                        })

            response_data = {"results": processed_results}

            # 添加详细的保存逻辑日志
            logger.info(f"Preparing to handle save_results (flag: {save_results})")
            if save_results:
                logger.info("Save results is True, attempting to save...")
                if processed_results:
                    filepath = self.save_search_results(
                        query, collection_id, processed_results
                    )
                    response_data["saved_filepath"] = filepath
                    logger.info(f"Results saved to: {filepath}")
                else:
                    logger.info("No results to save")

            logger.info(f"Processed results count: {len(processed_results)}")
            logger.info("Search completed successfully")
            return processed_results

        except Exception as e:
            logger.error(f"Error in Milvus search: {str(e)}")
            raise
        finally:
            try:
                connections.disconnect("default")
            except Exception as disconnect_err:
                logger.warning(
                    f"Error disconnecting from Milvus: {str(disconnect_err)}"
                )

    async def _search_chroma(
        self,
        query: str,
        collection_id: str,
        top_k: int = 3,
        threshold: float = 0.7,
        word_count_threshold: int = 20,
        save_results: bool = False,
    ) -> Dict[str, Any]:
        """
        在Chroma中执行向量搜索

        Args:
            query (str): 搜索查询文本
            collection_id (str): 要搜索的集合ID
            top_k (int): 返回的最大结果数量
            threshold (float): 相似度阈值
            word_count_threshold (int): 文本字数阈值
            save_results (bool): 是否保存搜索结果

        Returns:
            Dict[str, Any]: 包含搜索结果的字典
        """
        try:
            # 添加参数日志
            logger.info(f"Chroma search parameters:")
            logger.info(f"- Query: {query}")
            logger.info(f"- Collection ID: {collection_id}")
            logger.info(f"- Top K: {top_k}")
            logger.info(f"- Threshold: {threshold}")
            logger.info(f"- Word Count Threshold: {word_count_threshold}")
            logger.info(f"- Save Results: {save_results}")

            logger.info(
                f"Starting Chroma search with parameters - Collection: {collection_id}, Query: {query}, Top K: {top_k}"
            )

            # 创建Chroma客户端
            chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            # 获取collection
            logger.info(f"Loading Chroma collection: {collection_id}")
            collection = chroma_client.get_collection(collection_id)

            # 获取collection元数据信息
            col_metadata = collection.metadata
            logger.info(f"Collection metadata: {col_metadata}")

            # 使用collection中存储的配置创建查询向量
            logger.info("Creating query embedding")
            query_embedding = self.embedding_service.create_single_embedding(
                query,
                provider=col_metadata.get("embedding_provider", "openai"),
                model=col_metadata.get("embedding_model", "text-embedding-ada-002"),
            )

            # 执行搜索
            logger.info(f"Executing Chroma search with top_k: {top_k}")
            where_clause = (
                {"word_count": {"$gte": word_count_threshold}}
                if word_count_threshold > 0
                else None
            )

            results = collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where_clause
            )

            # 处理结果
            processed_results = []

            if results and "documents" in results and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                logger.info(f"Raw search results count: {len(docs)}")

                for i, (doc, metadata, distance) in enumerate(
                    zip(docs, metadatas, distances)
                ):
                    # 转换distance为score (1 - distance为相似度得分)
                    score = 1.0 - distance

                    if score >= threshold:
                        processed_results.append(
                            {
                                "text": doc,
                                "score": float(score),
                                "metadata": {
                                    "source": metadata.get("document_name", ""),
                                    "page": metadata.get("page_number", "0"),
                                    "chunk": metadata.get("chunk_id", 0),
                                    "total_chunks": metadata.get("total_chunks", 0),
                                    "page_range": metadata.get("page_range", ""),
                                    "embedding_provider": metadata.get(
                                        "embedding_provider", ""
                                    ),
                                    "embedding_model": metadata.get(
                                        "embedding_model", ""
                                    ),
                                    "embedding_timestamp": metadata.get(
                                        "embedding_timestamp", ""
                                    ),
                                },
                            }
                        )

            # 保存结果
            if save_results and processed_results:
                filepath = self.save_search_results(
                    query, collection_id, processed_results
                )
                logger.info(f"Results saved to: {filepath}")

            logger.info(f"Processed results count: {len(processed_results)}")
            logger.info("Chroma search completed successfully")
            return processed_results

        except Exception as e:
            logger.error(f"Error in Chroma search: {str(e)}")
            raise
