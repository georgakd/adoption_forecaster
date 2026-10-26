"""
RAG based feedback retrieval system.
This module implements a Retrieval-Augmented Generation (RAG) system to retrieve
relevant feedback for ML-generated projections. The system uses
semantic search to find similar feedback even when exact matches don't exist.
"""

from openai import OpenAI
import numpy as np
from typing import Optional, List
from adoption_forecaster.config.logging_config import get_logger


logger = get_logger(__name__)

##temporarily set cosine similarity score here, until we create a proper config file
cosine_score = 0.75


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2:
        logger.debug("[RAG] Empty vector(s) provided to cosine_similarity")
        return 0.0
    try:
        vec1, vec2 = np.array(vec1), np.array(vec2)

        # check that dimensions match
        if vec1.shape != vec2.shape:
            logger.warning(f"[RAG] Vector dimension mismarch: {vec1.shape} vs {vec2.shape}")
            return 0.0

        # check for nan
        if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
            logger.warning("[RAG] NaN values in vectors for cosine similarity")
            return 0.0

        dot = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        return dot / norms if norms > 0 else 0.0
    except Exception as e:
        logger.error(f"[RAG] Error calculating cosine similarity: {e}")


class FeedbackRAGRetriever:
    """Feedback Retriever Class"""

    def __init__(self, feedback_data: dict, openai_api_key: str = None):
        """
        Initialize RAG retriever for feedback
        Args:
            feedback_data: Dict ftom load_feedback()
            openai_api_key: OpenAI API key (e.g. from get_openai_api_key(config)).
        """
        if not feedback_data:
            logger.warning("[RAG] Empty feedback_data provided")
            self.feedback_data = {}
            self.embeddings_index = []
            return

        self.feedback_data = feedback_data

        try:
            self.client = OpenAI(api_key=openai_api_key)
            self.embeddings_index = []
            self._build_index()
        except Exception as e:
            logger.error(f"[RAG] Failed to initialize OpenAI Client: {e}")
            raise

    def _build_index(self):
        """Build embeddings index from feedback"""
        index_count = 0
        failed_count = 0
        for country, data in self.feedback_data.items():
            if "secondary" in data:
                feedback_item = data["secondary"]
                comment = feedback_item.get("comments", "")

                if comment and comment != "OK":
                    context = f"{country} secondary education {comment}"
                    embedding = self._get_embedding(context)

                    if embedding is None:
                        failed_count += 1
                        logger.warning("[RAG] failed to get embedding for {country} secondary feedback. Skipping...")
                        continue

                    self.embeddings_index.append(
                        {"country": country, "comment": comment, "embedding": embedding, "module": "education", "level": "secondary"}
                    )
                    index_count += 1
            if "age_gender" in data:
                for individual, feedback_item in data["age_gender"].items():
                    comment = feedback_item.get("comments", "")
                    if comment and comment != "OK":
                        context = f"{country} {individual} age gender {comment}"
                        embedding = self._get_embedding(context)

                        if embedding is None:
                            failed_count = +1
                            logger.warning("[RAG] failed to get embedding for {country} {individual} age_gender . Skipping...")
                            continue

                        self.embeddings_index.append(
                            {"country": country, "individual": individual, "comment": comment, "embedding": embedding, "module": "age_gender"}
                        )
                        index_count += 1
        logger.info(f"[RAG] Built embeddings index: {index_count} items indexed. {failed_count} failed ")
        if index_count == 0:
            logger.warning("[RAG] No embeddings were successfully indexed. Semantic search will not work.")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        if not text or not text.strip():
            logger.warning("[RAG] Empty text provided for embedding")
            return None
        try:
            response = self.client.embeddings.create(model="text-embedding-3-small", input=text)
            if not response or not response.data or len(response.data) == 0:
                logger.error("[RAG] Empty response from OpenAI embeddings API")
                return None

            embedding = response.data[0].embedding
            if not embedding or len(embedding) == 0:
                logger.error("[RAG] Empty embedding returned from API")
                return None

            return embedding

        except Exception as e:

            error_type = type(e).__name__
            error_msg = str(e)

            if "AuthenticationError" in error_type or "InvalidApiKey" in error_msg:
                logger.error("[RAG] OpenAI API authentication failed. Check OPENAI_API_KEY")
            elif "RateLimitError" in error_type or "rate_limit" in error_msg.lower():
                logger.warning("[RAG] OpenAI API rate limit exceeded. Retrying may help.")
            elif "APIConnectionError" in error_type or "Connection" in error_type:
                logger.warning("[RAG] Network error connecting to OpenAI API")
            elif "APIError" in error_type:
                logger.error(f"[RAG] OpenAI API error: {error_msg}")
            else:
                logger.error(f"[RAG] Unexpected error getting embedding: {error_type}: {error_msg}")

            return None

    def get_feedback_education(
        self, country: str, level: str, projection_value: float, previous_value: float = None  # e.g. secondary_2026, tertiary_2026
    ) -> Optional[str]:
        """
        Get feedback for education module
        IMPORTANT: Only Secondary gets feedback. Tertiary returns None.
        Tertiary is auto-adjusted after Secondary adjustment.
        Returns: Feedback text string or None
        """
        if not level.startswith("secondary"):
            return None
        # direct match - get secondary feedback for this country
        if country in self.feedback_data:
            if "secondary" in self.feedback_data[country]:
                feedback_item = self.feedback_data[country]["secondary"]
                comments = feedback_item.get("comments", "")

                if comments and comments != "OK":
                    logger.info(
                        f"[RAG] Direct match found for {country}: {comments[:100]}..."
                        if len(comments) > 100
                        else f"[RAG] Direct match found for {country}: {comments}"
                    )
                    return comments

        # semantic search using embeddings
        query = f"{country} secondary education projection {projection_value}%"

        if previous_value:
            query += f" change from {previous_value}%"

        return self._perform_semantic_search(query, entity_name=country, filter_module="education")

    def get_feedback_age_gender(
        self, country: str, individual: str, column: str, projection_value: float, previous_value: float = None
    ) -> Optional[str]:
        """
        Get feedback for age-gender module.
        Returns: Feedback text string or None
        """
        if country in self.feedback_data:
            if "age_gender" in self.feedback_data[country]:
                if individual in self.feedback_data[country]["age_gender"]:
                    feedback_item = self.feedback_data[country]["age_gender"][individual]
                    comments = feedback_item.get("comments", "")
                    if comments and comments != "OK":
                        logger.info(
                            f"[RAG] Direct match found for {country} {individual}: {comments[:100]}..."
                            if len(comments) > 100
                            else f"[RAG] Direct match found for {country} {individual}: {comments}"
                        )
                        return comments

        query = f"{country} {individual} age gender {column} projection {projection_value}"
        if previous_value:
            query += f"change from {previous_value}%"

        entity_name = f"{country} {individual}"
        return self._perform_semantic_search(query, entity_name=entity_name, filter_module="age_gender")

    def _perform_semantic_search(
        self,
        query: str,
        entity_name: str,
        filter_module: Optional[str] = None,
    ) -> Optional[str]:
        """
        Perform semantic search using embeddings.

        Args:
            query: Search query text
            entity_name: Name of entity being searched (for logging) - e.g., "Austria" or "Austria - Male_25-34"
            filter_module: Optional module filter ("education" or "age_gender"). If None, searches all items.

        Returns:
            Feedback comment string if match found above threshold, None otherwise
        """
        logger.debug(f"[RAG] Performing semantic search for: {query}")

        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning(f"[RAG] Failed to get embedding for query. Cannot perform semantic search for {entity_name}")
            return None

        if not self.embeddings_index:
            logger.debug("[RAG] Embeddings index is empty. Cannot perform semantic search.")
            return None

        best_match = None
        best_similarity = 0.0

        for item in self.embeddings_index:
            # Filter by module if specified
            if filter_module and item.get("module") != filter_module:
                continue

            # Skip items without embeddings
            if "embedding" not in item:
                logger.debug("[RAG] Item missing embedding, skipping")
                continue

            similarity = cosine_similarity(query_embedding, item["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = item

        # Return result if above threshold
        if best_match and best_similarity > cosine_score:
            comment = best_match.get("comment", "")
            match_country = best_match.get("country", "unknown")
            match_individual = best_match.get("individual", "")

            if match_individual:
                match_info = f"{match_country} {match_individual}"
            else:
                match_info = match_country

            log_msg = (
                f"[RAG] Semantic match found (similarity: {best_similarity:.3f}) for {entity_name} from {match_info}: {comment[:100]}..."
                if len(comment) > 100
                else f"[RAG] Semantic match found (similarity: {best_similarity:.3f}) for {entity_name} from {match_info}: {comment}"
            )
            logger.info(log_msg)
            return comment
        elif best_match:
            logger.debug(f"[RAG] Best match similarity ({best_similarity:.3f}) below threshold ({cosine_score}) for {entity_name}")
        else:
            logger.debug(f"[RAG] No semantic matches found for {entity_name}")

        return None
