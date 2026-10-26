from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, PointStruct, SparseVectorParams, 
    SparseVector, Prefetch, FusionQuery, Fusion
)
from typing import Iterable, List, Dict
from numpy.typing import NDArray
import numpy as np
from uuid import uuid4

from ...model.retrieved_document import RetrievedDocument
from ....config.logging_config import set_logger
from ....config.config_manager import get_config
from .asyncVectorStore import AsyncVectorStoreInterface
# from .VectorStore importAsyncVectorStoreInterface
  # Assuming you have an async interface

logger = set_logger(__name__, "ingestion.log")

class AsyncQdrantVectorStore(AsyncVectorStoreInterface):
    
    def __init__(self, client: AsyncQdrantClient):
        self.client = client
        self.ingestion_config = get_config()["ingestion"]

    async def collection_exists(self, collection_name: str) -> bool:
        exists = await self.client.collection_exists(collection_name)
        return exists

    async def create_collection(self):
        collection_name = self.ingestion_config["collection_name"]
        if not await self.collection_exists(collection_name):
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.ingestion_config["dense_vector_name"]: VectorParams(
                        size=self.ingestion_config["dense_vector_size"],
                        distance=self.ingestion_config["dense_vector_distance"]
                    )
                },
                sparse_vectors_config={
                    self.ingestion_config["sparse_vector_name"]: SparseVectorParams()
                }
            )


    async def save_documents(
        self,
        dense_embeddings: Iterable,
        sparse_embeddings: Iterable,
        collection_name: str,
        chunks: Iterable,
    ):
        points = []
        for dense, sparse, chunk in zip(dense_embeddings, sparse_embeddings, chunks):
            payload = {
                "content": chunk.page_content,
                **chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector={
                        self.ingestion_config["dense_vector_name"]: dense,
                        self.ingestion_config["sparse_vector_name"]: SparseVector(
                            indices=list(sparse.keys()),
                            values=list(sparse.values())
                        )
                    },
                    payload=payload,
                )
            )
        return await self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    async def similarity_dense_search(
        self, collection_name: str, query_embedding: NDArray[np.float32], limit: int = 25
    ) -> List[RetrievedDocument]:
        
        formatted_dense = (
            query_embedding.tolist() 
            if isinstance(query_embedding, np.ndarray) 
            else query_embedding
        )
        
        if formatted_dense is None:
            raise ValueError("`query_embedding` is required for dense search")
        
        result = await self.client.query_points(
            collection_name=collection_name,
            query=formatted_dense,
            using=self.ingestion_config["dense_vector_name"],
            with_payload=True,
            limit=limit,
        )
        
        return [
            RetrievedDocument(
                chunk_id=str(point.id),
                content=point.payload.get("content", ""),
                score=point.score,
                metadata=point.payload,
            ) for point in result.points
        ]

    async def similarity_sparse_search(
        self, collection_name: str, query_embedding: Dict[int, float], limit: int = 25
    ) -> List[RetrievedDocument]:
        
        if not query_embedding:
            raise ValueError("`query_embedding` is required for sparse search")
            
        formatted_sparse = SparseVector(
            indices=list(query_embedding.keys()),
            values=list(query_embedding.values()),
        )
        
        result = await self.client.query_points(
            collection_name=collection_name,
            query=formatted_sparse,
            using=self.ingestion_config["sparse_vector_name"],
            with_payload=True,
            limit=limit,
        )
        
        return [
            RetrievedDocument(
                chunk_id=str(point.id),
                content=point.payload.get("content", ""),
                score=point.score,
                metadata=point.payload,
            ) for point in result.points
        ]

    async def similarity_hybrid_search(
        self, 
        collection_name: str, 
        dense_query_embedding: NDArray[np.float32], 
        sparse_query_embedding: Dict[int, float], 
        limit: int = 25
    ) -> List[RetrievedDocument]:
        
        formatted_dense = (
            dense_query_embedding.tolist() 
            if isinstance(dense_query_embedding, np.ndarray) 
            else dense_query_embedding
        )
        
        formatted_sparse = SparseVector(
            indices=list(sparse_query_embedding.keys()),
            values=list(sparse_query_embedding.values()),
        ) if sparse_query_embedding else None

        if formatted_dense is None or formatted_sparse is None:
            raise ValueError("Both dense and sparse embeddings are required for hybrid search")

        result = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=formatted_dense,
                    using=self.ingestion_config["dense_vector_name"],
                    limit=limit * 2,
                ),
                Prefetch(
                    query=formatted_sparse,
                    using=self.ingestion_config["sparse_vector_name"],
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            with_payload=True,
            limit=limit,
        )

        return [
            RetrievedDocument(
                chunk_id=str(point.id),
                content=point.payload.get("content", ""),
                score=point.score,
                metadata=point.payload,
            ) for point in result.points
        ]

    async def close(self):
        return await self.client.close()
        
    async def delete_document(self):
        pass
    
    