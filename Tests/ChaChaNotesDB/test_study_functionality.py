# test_study_functionality.py
"""
Comprehensive tests for study-related functionality in ChaChaNotes_DB.
Tests flashcards, spaced repetition, learning paths, mindmaps, and study statistics.
"""

import pytest
import sqlite3
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Third-Party Imports
from hypothesis import strategies as st
from hypothesis import given, settings, HealthCheck

# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
    InputError,
    ConflictError
)


# --- Test Fixtures ---

@pytest.fixture
def client_id():
    """Provides a consistent client ID for tests."""
    return "test_study_client"


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "test_study_db.sqlite"


@pytest.fixture(scope="function")
def db_instance(db_path, client_id):
    """Creates a DB instance for each test, ensuring a fresh database."""
    current_db_path = Path(db_path)
    
    # Clean up any existing files
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(current_db_path) + suffix)
        if p.exists():
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    
    db = None
    try:
        db = CharactersRAGDB(current_db_path, client_id)
        yield db
    finally:
        if db:
            db.close_connection()
            # Cleanup after test
            for suffix in ["", "-wal", "-shm"]:
                p = Path(str(current_db_path) + suffix)
                if p.exists():
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass


@pytest.fixture
def mem_db_instance(client_id):
    """Creates an in-memory DB instance for faster tests."""
    db = CharactersRAGDB(":memory:", client_id)
    yield db
    db.close_connection()


@pytest.fixture
def sample_deck(db_instance):
    """Creates a sample deck and returns its ID."""
    deck_id = db_instance.create_deck("Test Deck", "A deck for testing")
    return deck_id


@pytest.fixture
def sample_flashcard(db_instance, sample_deck):
    """Creates a sample flashcard and returns its data."""
    card_data = {
        "deck_id": sample_deck,
        "front": "What is the capital of France?",
        "back": "Paris",
        "tags": "geography europe",
        "type": "basic"
    }
    card_id = db_instance.create_flashcard(card_data)
    return {
        "id": card_id,
        "deck_id": sample_deck,
        **card_data
    }


@pytest.fixture
def sample_learning_path(db_instance):
    """Creates a sample learning path and returns its ID."""
    path_id = db_instance.create_learning_path(
        "Python Programming",
        "Learn Python from basics to advanced"
    )
    return path_id


@pytest.fixture
def sample_mindmap(db_instance):
    """Creates a sample mindmap and returns its ID."""
    mindmap_id = db_instance.create_mindmap("Project Planning")
    return mindmap_id


# --- Helper Functions ---

def create_flashcard_data(deck_id, front="Test Question", back="Test Answer", 
                         tags="", type="basic"):
    """Helper to create flashcard test data."""
    return {
        "deck_id": deck_id,
        "front": front,
        "back": back,
        "tags": tags,
        "type": type
    }


# --- Test Classes ---

class TestFlashcardOperations:
    """Tests for flashcard CRUD operations."""
    
    def test_create_deck(self, db_instance):
        """Test creating a new flashcard deck."""
        deck_id = db_instance.create_deck("Spanish Vocabulary", "Learn Spanish words")
        assert deck_id is not None
        assert isinstance(deck_id, str)
        assert len(deck_id) == 36  # UUID length with dashes
    
    def test_create_deck_duplicate_name(self, db_instance):
        """Test that duplicate deck names raise an error."""
        db_instance.create_deck("Duplicate Deck", "First deck")
        with pytest.raises(sqlite3.IntegrityError):
            db_instance.create_deck("Duplicate Deck", "Second deck")
    
    def test_create_flashcard(self, db_instance, sample_deck):
        """Test creating a flashcard with valid data."""
        card_data = create_flashcard_data(
            sample_deck,
            "What is 2+2?",
            "4",
            "math basic"
        )
        card_id = db_instance.create_flashcard(card_data)
        
        assert card_id is not None
        assert isinstance(card_id, str)
        assert len(card_id) == 36
    
    def test_create_flashcard_missing_fields(self, db_instance, sample_deck):
        """Test that missing required fields raise an error."""
        # Missing 'back' field
        card_data = {
            "deck_id": sample_deck,
            "front": "Question only"
        }
        with pytest.raises(InputError, match="must have both front and back"):
            db_instance.create_flashcard(card_data)
        
        # Missing 'front' field
        card_data = {
            "deck_id": sample_deck,
            "back": "Answer only"
        }
        with pytest.raises(InputError, match="must have both front and back"):
            db_instance.create_flashcard(card_data)

    def test_create_flashcard_requires_active_deck(self, db_instance):
        missing_deck_id = str(uuid.uuid4())
        deleted_deck_id = db_instance.create_deck("Biology")
        db_instance.delete_deck(deleted_deck_id, expected_version=1)

        with pytest.raises(InputError, match="Deck .* not found"):
            db_instance.create_flashcard(create_flashcard_data(missing_deck_id, "ATP", "Energy"))

        with pytest.raises(InputError, match="Deck .* not found"):
            db_instance.create_flashcard(create_flashcard_data(deleted_deck_id, "ADP", "Spent energy"))

    def test_create_flashcard_recounts_active_cards_instead_of_incrementing_drift(self, db_instance):
        deck_id = db_instance.create_deck("Biology")
        first_card_id = db_instance.create_flashcard(create_flashcard_data(deck_id, "ATP", "Energy"))
        second_card_id = db_instance.create_flashcard(create_flashcard_data(deck_id, "ADP", "Lower energy"))

        with db_instance.transaction() as cursor:
            cursor.execute(
                "UPDATE flashcards SET is_deleted = 1 WHERE id = ?",
                (second_card_id,),
            )
            cursor.execute(
                "UPDATE decks SET card_count = 99 WHERE id = ?",
                (deck_id,),
            )

        third_card_id = db_instance.create_flashcard(create_flashcard_data(deck_id, "NADH", "Electron carrier"))
        deck = db_instance.get_deck(deck_id)
        first_card = db_instance.get_flashcard(first_card_id)
        second_card = db_instance.get_flashcard(second_card_id)
        third_card = db_instance.get_flashcard(third_card_id)

        assert deck is not None
        assert deck["card_count"] == 2
        assert first_card is not None
        assert second_card is None
        assert third_card is not None

    def test_get_flashcard(self, db_instance, sample_flashcard):
        """Test retrieving a flashcard by ID."""
        retrieved = db_instance.get_flashcard(sample_flashcard["id"])
        
        assert retrieved is not None
        assert retrieved["id"] == sample_flashcard["id"]
        assert retrieved["front"] == sample_flashcard["front"]
        assert retrieved["back"] == sample_flashcard["back"]
        assert retrieved["tags"] == sample_flashcard["tags"]
        assert retrieved["is_deleted"] == 0
        assert retrieved["interval"] == 0
        assert retrieved["repetitions"] == 0
        assert retrieved["ease_factor"] == 2.5
    
    def test_get_flashcard_not_found(self, db_instance):
        """Test retrieving a non-existent flashcard returns None."""
        fake_id = str(uuid.uuid4())
        result = db_instance.get_flashcard(fake_id)
        assert result is None
    
    def test_search_flashcards(self, db_instance, sample_deck):
        """Test searching flashcards using FTS."""
        # Create multiple flashcards
        db_instance.create_flashcard(create_flashcard_data(
            sample_deck, "Python functions", "def keyword", "programming python"
        ))
        db_instance.create_flashcard(create_flashcard_data(
            sample_deck, "JavaScript functions", "function keyword", "programming javascript"
        ))
        db_instance.create_flashcard(create_flashcard_data(
            sample_deck, "What is a variable?", "A named storage location", "programming basics"
        ))
        
        # Search for 'functions'
        results = db_instance.search_flashcards("functions")
        assert len(results) == 2
        
        # Search with deck filter
        results = db_instance.search_flashcards("functions", deck_id=sample_deck)
        assert len(results) == 2
        
        # Search for 'python'
        results = db_instance.search_flashcards("python")
        assert len(results) == 1
        assert "Python" in results[0]["front"]
    
    def test_get_due_flashcards(self, db_instance, sample_deck):
        """Test getting flashcards due for review."""
        # Create cards with different review states
        new_card_id = db_instance.create_flashcard(create_flashcard_data(
            sample_deck, "New card", "Never reviewed"
        ))
        
        # Get due cards (should include new cards)
        due_cards = db_instance.get_due_flashcards()
        assert len(due_cards) >= 1
        assert any(card["id"] == new_card_id for card in due_cards)
        
        # Test with deck filter
        due_cards = db_instance.get_due_flashcards(deck_id=sample_deck)
        assert all(card["deck_id"] == sample_deck for card in due_cards)
        
        # Test limit
        due_cards = db_instance.get_due_flashcards(limit=1)
        assert len(due_cards) <= 1

    def test_get_deck_returns_created_deck_record(self, db_instance, sample_flashcard):
        """Decks should be fetchable by ID for explicit deck selection flows."""
        deck = db_instance.get_deck(sample_flashcard["deck_id"])

        assert deck is not None
        assert deck["id"] == sample_flashcard["deck_id"]
        assert deck["name"] == "Test Deck"
        assert deck["card_count"] == 1

    def test_update_deck_updates_metadata_and_version(self, db_instance, sample_flashcard):
        updated = db_instance.update_deck(
            sample_flashcard["deck_id"],
            name="Updated Deck",
            description="Updated description",
            expected_version=1,
        )
        deck = db_instance.get_deck(sample_flashcard["deck_id"])

        assert updated is True
        assert deck is not None
        assert deck["name"] == "Updated Deck"
        assert deck["description"] == "Updated description"
        assert deck["version"] == 2
        assert deck["card_count"] == 1

    def test_update_deck_rejects_stale_version_without_mutation(self, db_instance, sample_flashcard):
        with pytest.raises(ConflictError, match="Version mismatch updating deck"):
            db_instance.update_deck(sample_flashcard["deck_id"], name="Stale", expected_version=0)

        deck = db_instance.get_deck(sample_flashcard["deck_id"])

        assert deck is not None
        assert deck["name"] == "Test Deck"
        assert deck["version"] == 1

    def test_update_deck_rejects_blank_name_without_mutation(self, db_instance, sample_flashcard):
        with pytest.raises(InputError, match="Deck name cannot be blank"):
            db_instance.update_deck(sample_flashcard["deck_id"], name="   ", expected_version=1)

        deck = db_instance.get_deck(sample_flashcard["deck_id"])

        assert deck is not None
        assert deck["name"] == "Test Deck"
        assert deck["version"] == 1

    def test_delete_flashcard_hides_card_and_recounts_deck(self, db_instance, sample_flashcard):
        deck_before_delete = db_instance.get_deck(sample_flashcard["deck_id"])

        deleted = db_instance.delete_flashcard(sample_flashcard["id"], expected_version=1)
        deck_after_delete = db_instance.get_deck(sample_flashcard["deck_id"])

        assert deck_before_delete is not None
        assert deck_before_delete["card_count"] == 1
        assert deleted is True
        assert db_instance.get_flashcard(sample_flashcard["id"]) is None
        assert deck_after_delete is not None
        assert deck_after_delete["card_count"] == 0

    def test_move_flashcard_changes_deck_and_recounts_both_decks(self, db_instance):
        source = db_instance.create_deck("Biology")
        target = db_instance.create_deck("Chemistry")
        card_id = db_instance.create_flashcard(create_flashcard_data(source, "ATP", "Energy"))

        moved = db_instance.move_flashcard(card_id, target, expected_version=1)
        moved_card = db_instance.get_flashcard(card_id)
        source_deck = db_instance.get_deck(source)
        target_deck = db_instance.get_deck(target)

        assert moved is True
        assert moved_card is not None
        assert moved_card["deck_id"] == target
        assert source_deck is not None
        assert source_deck["card_count"] == 0
        assert target_deck is not None
        assert target_deck["card_count"] == 1

    def test_move_flashcard_rejects_missing_or_deleted_target_deck(self, db_instance):
        source = db_instance.create_deck("Biology")
        deleted_target = db_instance.create_deck("Chemistry")
        card_id = db_instance.create_flashcard(create_flashcard_data(source, "ATP", "Energy"))
        db_instance.delete_deck(deleted_target, expected_version=1)

        with pytest.raises(InputError, match="Target deck .* not found"):
            db_instance.move_flashcard(card_id, str(uuid.uuid4()), expected_version=1)

        with pytest.raises(InputError, match="Target deck .* not found"):
            db_instance.move_flashcard(card_id, deleted_target, expected_version=1)

    def test_delete_flashcard_stale_version_raises_conflict_without_mutation(self, db_instance, sample_flashcard):
        deck_before = db_instance.get_deck(sample_flashcard["deck_id"])

        with pytest.raises(ConflictError, match="Version mismatch deleting flashcard"):
            db_instance.delete_flashcard(sample_flashcard["id"], expected_version=0)

        card_after = db_instance.get_flashcard(sample_flashcard["id"])
        deck_after = db_instance.get_deck(sample_flashcard["deck_id"])

        assert deck_before is not None
        assert card_after is not None
        assert card_after["is_deleted"] == 0
        assert deck_after is not None
        assert deck_after["card_count"] == deck_before["card_count"] == 1

    def test_move_flashcard_stale_version_raises_conflict_without_mutation(self, db_instance):
        source = db_instance.create_deck("Biology")
        target = db_instance.create_deck("Chemistry")
        card_id = db_instance.create_flashcard(create_flashcard_data(source, "ATP", "Energy"))

        with pytest.raises(ConflictError, match="Version mismatch moving flashcard"):
            db_instance.move_flashcard(card_id, target, expected_version=0)

        moved_card = db_instance.get_flashcard(card_id)
        source_deck = db_instance.get_deck(source)
        target_deck = db_instance.get_deck(target)

        assert moved_card is not None
        assert moved_card["deck_id"] == source
        assert source_deck is not None
        assert source_deck["card_count"] == 1
        assert target_deck is not None
        assert target_deck["card_count"] == 0

    def test_delete_deck_tombstones_name_hides_children_and_allows_recreate(self, db_instance):
        deck_id = db_instance.create_deck("Biology")
        card_id = db_instance.create_flashcard(create_flashcard_data(deck_id, "ATP", "Energy"))

        deleted = db_instance.delete_deck(deck_id, expected_version=1)
        replacement_deck_id = db_instance.create_deck("Biology")

        assert deleted is True
        assert db_instance.get_deck(deck_id) is None
        assert replacement_deck_id is not None
        assert replacement_deck_id != deck_id
        assert db_instance.get_flashcard(card_id) is None
        assert db_instance.list_flashcards(deck_id=deck_id, limit=10, offset=0) == []
        assert db_instance.get_due_flashcards(deck_id=deck_id, limit=10) == []

    def test_delete_deck_stale_version_raises_conflict_without_mutation(self, db_instance):
        deck_id = db_instance.create_deck("Biology")
        card_id = db_instance.create_flashcard(create_flashcard_data(deck_id, "ATP", "Energy"))

        with pytest.raises(ConflictError, match="Version mismatch deleting deck"):
            db_instance.delete_deck(deck_id, expected_version=0)

        deck = db_instance.get_deck(deck_id)
        card = db_instance.get_flashcard(card_id)

        assert deck is not None
        assert deck["name"] == "Biology"
        assert deck["card_count"] == 1
        assert card is not None
        assert card["deck_id"] == deck_id
        assert card["is_deleted"] == 0

    def test_delete_deck_renames_deleted_row_to_deterministic_tombstone(self, db_instance):
        deck_id = db_instance.create_deck("Biology")
        db_instance.create_flashcard(create_flashcard_data(deck_id, "ATP", "Energy"))

        deleted = db_instance.delete_deck(deck_id, expected_version=1)

        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT id, name, is_deleted, card_count FROM decks WHERE id = ?",
                (deck_id,),
            )
            deleted_row = cursor.fetchone()

        assert deleted is True
        assert deleted_row is not None
        assert deleted_row["id"] == deck_id
        assert deleted_row["name"] == f"__deleted_deck__:{deck_id}"
        assert deleted_row["is_deleted"] == 1
        assert deleted_row["card_count"] == 0

    def test_deleted_deck_visibility_filters_active_child_cards(self, db_instance):
        deck_id = db_instance.create_deck("Biology")
        card_id = db_instance.create_flashcard(create_flashcard_data(deck_id, "ATP", "Energy"))

        with db_instance.transaction() as cursor:
            cursor.execute(
                "UPDATE decks SET is_deleted = 1 WHERE id = ?",
                (deck_id,),
            )
            cursor.execute(
                "SELECT is_deleted FROM flashcards WHERE id = ?",
                (card_id,),
            )
            card_row = cursor.fetchone()

        list_results = db_instance.list_flashcards(deck_id=deck_id, limit=10, offset=0)
        due_results = db_instance.get_due_flashcards(deck_id=deck_id, limit=10)
        visible_card = db_instance.get_flashcard(card_id)

        assert card_row is not None
        assert card_row["is_deleted"] == 0
        assert visible_card is not None
        assert visible_card["id"] == card_id
        assert list_results == []
        assert due_results == []


class TestQuizOperations:
    """Tests for quiz CRUD and attempt flows."""

    def test_create_quiz_and_get_quiz(self, db_instance):
        quiz_id = db_instance.create_quiz(
            name="Geography Review",
            description="Capitals",
            time_limit_seconds=300,
            passing_score=70,
        )

        quiz = db_instance.get_quiz(quiz_id)

        assert quiz is not None
        assert quiz["id"] == quiz_id
        assert quiz["name"] == "Geography Review"
        assert quiz["total_questions"] == 0
        assert quiz["time_limit_seconds"] == 300

    def test_list_quizzes_blank_search_behaves_like_list(self, db_instance):
        first_id = db_instance.create_quiz(name="Geography Review")
        second_id = db_instance.create_quiz(name="History Review")

        listed = db_instance.list_quizzes(q="", limit=10, offset=0)

        listed_ids = {item["id"] for item in listed["items"]}
        assert first_id in listed_ids
        assert second_id in listed_ids
        assert listed["count"] >= 2

    def test_create_question_and_list_questions(self, db_instance):
        quiz_id = db_instance.create_quiz(name="Geography Review")
        question_id = db_instance.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="The capital of France is ____.",
            correct_answer="Paris",
            explanation="Paris is the capital city.",
            points=2,
        )

        question = db_instance.get_question(question_id)
        questions = db_instance.list_questions(quiz_id, include_answers=True, limit=10, offset=0)

        assert question is not None
        assert question["id"] == question_id
        assert question["correct_answer"] == "Paris"
        assert questions["count"] == 1
        assert questions["items"][0]["question_text"] == "The capital of France is ____."

    def test_start_and_submit_quiz_attempt(self, db_instance):
        quiz_id = db_instance.create_quiz(name="Geography Review")
        question_id = db_instance.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="The capital of France is ____.",
            correct_answer="Paris",
            explanation="Paris is the capital city.",
            points=2,
        )

        started = db_instance.start_attempt(quiz_id)
        submitted = db_instance.submit_attempt(
            started["id"],
            [{"question_id": question_id, "user_answer": "Paris", "time_spent_ms": 1200}],
        )
        listed = db_instance.list_attempts(quiz_id=quiz_id, limit=10, offset=0)
        loaded = db_instance.get_attempt(started["id"], include_questions=True, include_answers=True)

        assert started["completed_at"] is None
        assert started["questions"][0]["question_text"] == "The capital of France is ____."
        assert submitted["score"] == 2
        assert submitted["answers"][0]["is_correct"] is True
        assert listed["count"] == 1
        assert loaded["id"] == started["id"]
        assert loaded["questions"][0]["id"] == question_id

    def test_delete_quiz_hides_quiz_and_questions_from_default_reads(self, db_instance):
        quiz_id = db_instance.create_quiz(name="Geography Review")
        question_id = db_instance.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="The capital of France is ____.",
            correct_answer="Paris",
            explanation="Paris is the capital city.",
            points=2,
        )

        deleted = db_instance.delete_quiz(quiz_id)
        listed_quizzes = db_instance.list_quizzes(limit=10, offset=0)
        listed_questions = db_instance.list_questions(quiz_id, include_answers=True, limit=10, offset=0)

        assert deleted is True
        assert db_instance.get_quiz(quiz_id) is None
        assert db_instance.get_question(question_id) is None
        assert quiz_id not in {item["id"] for item in listed_quizzes["items"]}
        assert listed_questions["count"] == 0

    def test_delete_question_recounts_quiz_and_removes_it_from_lists(self, db_instance):
        quiz_id = db_instance.create_quiz(name="Geography Review")
        first_question_id = db_instance.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="The capital of France is ____.",
            correct_answer="Paris",
            explanation="Paris is the capital city.",
            points=2,
            order_index=0,
        )
        second_question_id = db_instance.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="The capital of Spain is ____.",
            correct_answer="Madrid",
            explanation="Madrid is the capital city.",
            points=2,
            order_index=1,
        )

        deleted = db_instance.delete_question(first_question_id)
        quiz = db_instance.get_quiz(quiz_id)
        listed_questions = db_instance.list_questions(quiz_id, include_answers=True, limit=10, offset=0)

        assert deleted is True
        assert db_instance.get_question(first_question_id) is None
        assert quiz is not None
        assert quiz["total_questions"] == 1
        assert [item["id"] for item in listed_questions["items"]] == [second_question_id]

    def test_list_decks_returns_non_deleted_decks_with_paging(self, db_instance):
        """Deck listing should power the flashcards deck selector without implicit defaults."""
        first = db_instance.create_deck("Biology", "Cells")
        second = db_instance.create_deck("Chemistry", "Atoms")

        page = db_instance.list_decks(limit=1, offset=0)
        next_page = db_instance.list_decks(limit=1, offset=1)

        assert len(page) == 1
        assert page[0]["id"] in {first, second}
        assert len(next_page) == 1
        assert next_page[0]["id"] != page[0]["id"]

    def test_list_flashcards_filters_by_deck_and_treats_blank_query_as_list(self, db_instance):
        """Blank search should not invoke FTS-only semantics for deck scoped lists."""
        biology = db_instance.create_deck("Biology", "Cells")
        chemistry = db_instance.create_deck("Chemistry", "Atoms")
        biology_card = db_instance.create_flashcard(create_flashcard_data(biology, "ATP", "Energy", "biology"))
        db_instance.create_flashcard(create_flashcard_data(chemistry, "H2O", "Water", "chemistry"))

        blank_query_results = db_instance.list_flashcards(deck_id=biology, q="   ", limit=10, offset=0)
        token_results = db_instance.list_flashcards(deck_id=biology, q="ATP", limit=10, offset=0)

        assert [row["id"] for row in blank_query_results] == [biology_card]
        assert [row["id"] for row in token_results] == [biology_card]


class TestSpacedRepetition:
    """Tests for the spaced repetition algorithm (SM-2)."""
    
    def test_update_flashcard_review_rating_0(self, db_instance, sample_flashcard):
        """Test review with rating 0 (complete blackout)."""
        card_id = sample_flashcard["id"]
        
        # Review with rating 0
        db_instance.update_flashcard_review(card_id, 0)
        
        # Check updated values
        card = db_instance.get_flashcard(card_id)
        assert card["repetitions"] == 0  # Reset to 0
        assert card["interval"] == 1  # Reset to 1 day
        assert card["last_review"] is not None
        assert card["next_review"] is not None
        
        # Verify review history was created
        with db_instance.transaction() as cursor:
            cursor.execute("SELECT * FROM review_history WHERE flashcard_id = ?", (card_id,))
            history = cursor.fetchone()
            assert history is not None
            assert history["rating"] == 0
            assert history["interval_after"] == 1
    
    def test_update_flashcard_review_rating_3(self, db_instance, sample_flashcard):
        """Test review with rating 3 (correct but difficult)."""
        card_id = sample_flashcard["id"]
        
        # First review
        db_instance.update_flashcard_review(card_id, 3)
        card = db_instance.get_flashcard(card_id)
        assert card["repetitions"] == 1
        assert card["interval"] == 1  # First interval
        
        # Second review
        db_instance.update_flashcard_review(card_id, 3)
        card = db_instance.get_flashcard(card_id)
        assert card["repetitions"] == 2
        assert card["interval"] == 6  # Second interval
        
        # Third review
        db_instance.update_flashcard_review(card_id, 3)
        card = db_instance.get_flashcard(card_id)
        assert card["repetitions"] == 3
        assert card["interval"] > 6  # Should be multiplied by ease factor
    
    def test_update_flashcard_review_rating_5(self, db_instance, sample_flashcard):
        """Test review with rating 5 (perfect response)."""
        card_id = sample_flashcard["id"]
        
        # Review with perfect rating
        db_instance.update_flashcard_review(card_id, 5)
        
        card = db_instance.get_flashcard(card_id)
        assert card["repetitions"] == 1
        assert card["interval"] == 1
        assert card["ease_factor"] > 2.5  # Should increase
    
    def test_ease_factor_adjustment(self, db_instance, sample_flashcard):
        """Test that ease factor adjusts correctly based on ratings."""
        card_id = sample_flashcard["id"]
        initial_ease = 2.5
        
        # Rating 5 should increase ease factor
        db_instance.update_flashcard_review(card_id, 5)
        card = db_instance.get_flashcard(card_id)
        assert card["ease_factor"] > initial_ease
        
        # Create another card for testing decrease
        card2_data = create_flashcard_data(
            sample_flashcard["deck_id"],
            "Test ease decrease",
            "Answer"
        )
        card2_id = db_instance.create_flashcard(card2_data)
        
        # Rating 1 should decrease ease factor
        db_instance.update_flashcard_review(card2_id, 1)
        card2 = db_instance.get_flashcard(card2_id)
        assert card2["ease_factor"] < initial_ease
        assert card2["ease_factor"] >= 1.3  # Should not go below minimum
    
    def test_invalid_rating(self, db_instance, sample_flashcard):
        """Test that invalid ratings raise an error."""
        with pytest.raises(InputError, match="Rating must be between 0 and 5"):
            db_instance.update_flashcard_review(sample_flashcard["id"], -1)
        
        with pytest.raises(InputError, match="Rating must be between 0 and 5"):
            db_instance.update_flashcard_review(sample_flashcard["id"], 6)
    
    def test_next_review_date_calculation(self, db_instance, sample_flashcard):
        """Test that next review date is calculated correctly."""
        card_id = sample_flashcard["id"]
        
        # Review the card
        db_instance.update_flashcard_review(card_id, 4)
        
        card = db_instance.get_flashcard(card_id)
        assert card["next_review"] is not None, "next_review should be set after review"
        
        # Check if next_review is already a datetime object
        if isinstance(card["next_review"], datetime):
            next_review = card["next_review"]
            # Ensure it has timezone info
            if next_review.tzinfo is None:
                next_review = next_review.replace(tzinfo=timezone.utc)
        else:
            # It's a string, parse it
            if card["next_review"].endswith('Z'):
                next_review = datetime.fromisoformat(card["next_review"].replace('Z', '+00:00'))
            else:
                next_review = datetime.fromisoformat(card["next_review"])
        
        now = datetime.now(timezone.utc)
        
        # Next review should be approximately 1 day from now
        time_diff = next_review - now
        # time_diff.days is an integer, so check total seconds instead
        assert 0.9 * 24 * 3600 < time_diff.total_seconds() < 1.1 * 24 * 3600  # Allow small variance


class TestLearningPaths:
    """Tests for learning paths and topics."""
    
    def test_create_learning_path(self, db_instance):
        """Test creating a new learning path."""
        path_id = db_instance.create_learning_path(
            "Data Science Fundamentals",
            "Learn the basics of data science"
        )
        assert path_id is not None
        assert isinstance(path_id, str)
        assert len(path_id) == 36
    
    def test_create_topic(self, db_instance, sample_learning_path):
        """Test creating a topic within a learning path."""
        topic_data = {
            "path_id": sample_learning_path,
            "title": "Introduction to Variables",
            "content": "Variables are containers for storing data values.",
            "topic_order": 1
        }
        topic_id = db_instance.create_topic(topic_data)
        
        assert topic_id is not None
        assert isinstance(topic_id, str)
        assert len(topic_id) == 36
    
    def test_create_topic_with_parent(self, db_instance, sample_learning_path):
        """Test creating nested topics."""
        # Create parent topic
        parent_data = {
            "path_id": sample_learning_path,
            "title": "Chapter 1: Basics",
            "content": "Introduction chapter"
        }
        parent_id = db_instance.create_topic(parent_data)
        
        # Create child topic
        child_data = {
            "path_id": sample_learning_path,
            "parent_id": parent_id,
            "title": "1.1 Getting Started",
            "content": "First steps"
        }
        child_id = db_instance.create_topic(child_data)
        
        assert child_id is not None
        
        # Verify parent-child relationship
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT parent_id FROM topics WHERE id = ?",
                (child_id,)
            )
            result = cursor.fetchone()
            assert result["parent_id"] == parent_id
    
    def test_update_topic_progress(self, db_instance, sample_learning_path):
        """Test updating topic progress."""
        topic_data = {
            "path_id": sample_learning_path,
            "title": "Test Topic"
        }
        topic_id = db_instance.create_topic(topic_data)
        
        # Update progress to 50%
        db_instance.update_topic_progress(topic_id, 0.5)
        
        # Verify update
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT progress, status FROM topics WHERE id = ?",
                (topic_id,)
            )
            result = cursor.fetchone()
            assert result["progress"] == 0.5
            assert result["status"] == "not_started"  # Status not changed
        
        # Update progress with status
        db_instance.update_topic_progress(topic_id, 1.0, "completed")
        
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT progress, status FROM topics WHERE id = ?",
                (topic_id,)
            )
            result = cursor.fetchone()
            assert result["progress"] == 1.0
            assert result["status"] == "completed"
    
    def test_invalid_progress_value(self, db_instance, sample_learning_path):
        """Test that invalid progress values raise an error."""
        topic_data = {"path_id": sample_learning_path, "title": "Test"}
        topic_id = db_instance.create_topic(topic_data)
        
        with pytest.raises(InputError, match="Progress must be between 0 and 1"):
            db_instance.update_topic_progress(topic_id, -0.1)
        
        with pytest.raises(InputError, match="Progress must be between 0 and 1"):
            db_instance.update_topic_progress(topic_id, 1.1)
    
    def test_invalid_status_value(self, db_instance, sample_learning_path):
        """Test that invalid status values raise an error."""
        topic_data = {"path_id": sample_learning_path, "title": "Test"}
        topic_id = db_instance.create_topic(topic_data)
        
        with pytest.raises(InputError, match="Invalid topic status"):
            db_instance.update_topic_progress(topic_id, 0.5, "invalid_status")


class TestMindmaps:
    """Tests for mindmap functionality."""
    
    def test_create_mindmap(self, db_instance):
        """Test creating a new mindmap."""
        mindmap_id = db_instance.create_mindmap("Study Plan")
        assert mindmap_id is not None
        assert isinstance(mindmap_id, str)
        assert len(mindmap_id) == 36
    
    def test_add_mindmap_node(self, db_instance, sample_mindmap):
        """Test adding a root node to a mindmap."""
        node_id = db_instance.add_mindmap_node(
            sample_mindmap,
            "Central Idea",
            position=(0, 0)
        )
        
        assert node_id is not None
        assert isinstance(node_id, str)
        
        # Verify node was created
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT * FROM mindmap_nodes WHERE id = ?",
                (node_id,)
            )
            node = cursor.fetchone()
            assert node["text"] == "Central Idea"
            assert node["position_x"] == 0
            assert node["position_y"] == 0
            assert node["parent_id"] is None
    
    def test_add_mindmap_node_with_parent(self, db_instance, sample_mindmap):
        """Test adding child nodes to a mindmap."""
        # Create root node
        root_id = db_instance.add_mindmap_node(
            sample_mindmap,
            "Main Topic"
        )
        
        # Add child nodes
        child1_id = db_instance.add_mindmap_node(
            sample_mindmap,
            "Subtopic 1",
            parent_id=root_id,
            position=(100, 50)
        )
        child2_id = db_instance.add_mindmap_node(
            sample_mindmap,
            "Subtopic 2",
            parent_id=root_id,
            position=(100, -50)
        )
        
        # Verify parent-child relationships
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM mindmap_nodes WHERE parent_id = ?",
                (root_id,)
            )
            result = cursor.fetchone()
            assert result["count"] == 2


class TestStudyStatistics:
    """Tests for study statistics functionality."""
    
    def test_get_study_stats_empty(self, db_instance):
        """Test getting stats from empty database."""
        stats = db_instance.get_study_stats(days=30)
        
        assert stats is not None
        assert "reviews" in stats
        assert "topics" in stats
        assert "sessions" in stats
        
        # Empty database should have zero counts
        assert stats["reviews"]["total_reviews"] == 0
        assert stats["topics"]["topics_completed"] == 0
        assert stats["sessions"]["total_sessions"] == 0
    
    def test_get_study_stats_with_data(self, db_instance, sample_deck):
        """Test getting stats with actual study data."""
        # Create and review some flashcards
        card1_id = db_instance.create_flashcard(create_flashcard_data(
            sample_deck, "Q1", "A1"
        ))
        card2_id = db_instance.create_flashcard(create_flashcard_data(
            sample_deck, "Q2", "A2"
        ))
        
        # Add some reviews
        db_instance.update_flashcard_review(card1_id, 4)
        db_instance.update_flashcard_review(card2_id, 3)
        db_instance.update_flashcard_review(card1_id, 5)
        
        # Create a completed topic
        path_id = db_instance.create_learning_path("Test Path")
        topic_id = db_instance.create_topic({
            "path_id": path_id,
            "title": "Test Topic"
        })
        db_instance.update_topic_progress(topic_id, 1.0, "completed")
        
        # Get stats
        stats = db_instance.get_study_stats(days=30)
        
        assert stats["reviews"]["total_reviews"] == 3
        assert stats["reviews"]["avg_rating"] == pytest.approx(4.0)  # (4+3+5)/3
        assert stats["topics"]["topics_completed"] == 1
    
    def test_study_stats_date_filtering(self, db_instance, sample_deck):
        """Test that date filtering works correctly."""
        # Create a flashcard
        card_id = db_instance.create_flashcard(create_flashcard_data(
            sample_deck, "Q", "A"
        ))
        
        # Add a recent review
        db_instance.update_flashcard_review(card_id, 4)
        
        # Stats for last 30 days should include the review
        stats_30 = db_instance.get_study_stats(days=30)
        assert stats_30["reviews"]["total_reviews"] == 1
        
        # Stats for 0 days should exclude it
        stats_0 = db_instance.get_study_stats(days=0)
        assert stats_0["reviews"]["total_reviews"] == 0


class TestStudyIntegration:
    """Integration tests for study functionality."""
    
    def test_full_study_workflow(self, db_instance):
        """Test a complete study workflow from deck creation to review."""
        # 1. Create a deck
        deck_id = db_instance.create_deck("Integration Test Deck")
        
        # 2. Add multiple flashcards
        card_ids = []
        for i in range(5):
            card_data = create_flashcard_data(
                deck_id,
                f"Question {i+1}",
                f"Answer {i+1}",
                "integration test"
            )
            card_id = db_instance.create_flashcard(card_data)
            card_ids.append(card_id)
        
        # 3. Get due cards
        due_cards = db_instance.get_due_flashcards(deck_id=deck_id)
        assert len(due_cards) == 5
        
        # 4. Review some cards
        db_instance.update_flashcard_review(card_ids[0], 5)  # Perfect
        db_instance.update_flashcard_review(card_ids[1], 3)  # Difficult
        db_instance.update_flashcard_review(card_ids[2], 0)  # Forgot
        
        # 5. Check updated due cards
        due_cards = db_instance.get_due_flashcards(deck_id=deck_id)
        # Cards 3 and 4 haven't been reviewed yet
        assert len([c for c in due_cards if c["next_review"] is None]) == 2
        
        # 6. Search for cards
        search_results = db_instance.search_flashcards("Question", deck_id=deck_id)
        assert len(search_results) == 5
        
        # 7. Get statistics
        stats = db_instance.get_study_stats()
        assert stats["reviews"]["total_reviews"] == 3
        assert stats["reviews"]["avg_rating"] == pytest.approx(2.67, rel=0.01)  # (5+3+0)/3
    
    def test_learning_path_workflow(self, db_instance):
        """Test complete learning path workflow."""
        # 1. Create learning path
        path_id = db_instance.create_learning_path(
            "Complete Python Course",
            "From zero to hero"
        )
        
        # 2. Create chapter topics
        chapter_ids = []
        for i in range(3):
            topic_id = db_instance.create_topic({
                "path_id": path_id,
                "title": f"Chapter {i+1}",
                "topic_order": i
            })
            chapter_ids.append(topic_id)
        
        # 3. Add subtopics to first chapter
        for i in range(3):
            db_instance.create_topic({
                "path_id": path_id,
                "parent_id": chapter_ids[0],
                "title": f"Section 1.{i+1}",
                "topic_order": i
            })
        
        # 4. Update progress on topics
        db_instance.update_topic_progress(chapter_ids[0], 1.0, "completed")
        db_instance.update_topic_progress(chapter_ids[1], 0.5, "in_progress")
        
        # 5. Get stats
        stats = db_instance.get_study_stats()
        assert stats["topics"]["topics_completed"] == 1
    
    def test_concurrent_operations(self, db_instance):
        """Test that concurrent operations work correctly with retry logic."""
        import threading
        import time
        import sqlite3
        
        deck_id = db_instance.create_deck("Concurrent Test Deck")
        results = []
        errors = []
        lock = threading.Lock()
        
        def create_cards(start_idx, count):
            for i in range(count):
                card_data = create_flashcard_data(
                    deck_id,
                    f"Q{start_idx + i}",
                    f"A{start_idx + i}"
                )
                
                # Retry logic for database locks
                max_retries = 5
                retry_delay = 0.1
                
                for attempt in range(max_retries):
                    try:
                        card_id = db_instance.create_flashcard(card_data)
                        with lock:
                            results.append(card_id)
                        break
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            with lock:
                                errors.append(e)
                            break
                    except Exception as e:
                        with lock:
                            errors.append(e)
                        break
        
        # Create threads
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=create_cards,
                args=(i * 10, 10)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results - with retry logic, we should succeed
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(results) == 30, f"Expected 30 results, got {len(results)}"
        assert len(set(results)) == 30, "All IDs should be unique"


# Property-based tests using Hypothesis
class TestStudyProperties:
    """Property-based tests for study functionality."""
    
    @given(
        front=st.text(min_size=1, max_size=1000),
        back=st.text(min_size=1, max_size=1000),
        tags=st.text(max_size=500),
        card_type=st.sampled_from(["basic", "cloze", "reverse"])
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_flashcard_properties(self, mem_db_instance, front, back, tags, card_type):
        """Test that any valid flashcard data can be created and retrieved."""
        # Create deck first with unique name using UUID
        deck_id = mem_db_instance.create_deck(f"Property Test Deck {uuid.uuid4()}")
        
        # Create flashcard
        card_data = {
            "deck_id": deck_id,
            "front": front,
            "back": back,
            "tags": tags,
            "type": card_type
        }
        card_id = mem_db_instance.create_flashcard(card_data)
        
        # Retrieve and verify
        retrieved = mem_db_instance.get_flashcard(card_id)
        assert retrieved["front"] == front
        assert retrieved["back"] == back
        assert retrieved["tags"] == tags
        assert retrieved["type"] == card_type
    
    @given(rating=st.integers(min_value=0, max_value=5))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_spaced_repetition_properties(self, mem_db_instance, rating):
        """Test that any valid rating produces valid interval and ease factor."""
        # Setup with unique deck name
        deck_id = mem_db_instance.create_deck(f"SR Test Deck {uuid.uuid4()}")
        card_id = mem_db_instance.create_flashcard(create_flashcard_data(deck_id))
        
        # Review
        mem_db_instance.update_flashcard_review(card_id, rating)
        
        # Verify invariants
        card = mem_db_instance.get_flashcard(card_id)
        assert card["interval"] >= 1
        assert card["ease_factor"] >= 1.3
        assert card["repetitions"] >= 0
        assert card["last_review"] is not None
        assert card["next_review"] is not None
        
        if rating < 3:
            assert card["repetitions"] == 0
            assert card["interval"] == 1
    
    @given(progress=st.floats(min_value=0.0, max_value=1.0))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_progress_properties(self, mem_db_instance, progress):
        """Test that any valid progress value can be set."""
        path_id = mem_db_instance.create_learning_path("Progress Test")
        topic_id = mem_db_instance.create_topic({
            "path_id": path_id,
            "title": "Test Topic"
        })
        
        # Update progress
        mem_db_instance.update_topic_progress(topic_id, progress)
        
        # Verify
        with mem_db_instance.transaction() as cursor:
            cursor.execute("SELECT progress FROM topics WHERE id = ?", (topic_id,))
            result = cursor.fetchone()
            assert result["progress"] == pytest.approx(progress)


class TestSchemaMigration:
    """Test that the schema migration from v10 to v11 works correctly."""
    
    def test_migration_creates_all_tables(self, db_instance):
        """Test that all study tables are created after migration."""
        with db_instance.transaction() as cursor:
            # Check that all tables exist
            tables = [
                "learning_paths", "topics", "decks", "flashcards",
                "review_history", "mindmaps", "mindmap_nodes", "study_sessions",
                "quizzes", "quiz_questions", "quiz_attempts"
            ]
        
            for table in tables:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,)
                )
                result = cursor.fetchone()
                assert result is not None, f"Table {table} should exist"
    
    def test_migration_creates_fts_tables(self, db_instance):
        """Test that FTS tables are created."""
        with db_instance.transaction() as cursor:
            fts_tables = ["topics_fts", "flashcards_fts", "mindmap_nodes_fts"]

            for table in fts_tables:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,)
                )
                result = cursor.fetchone()
                assert result is not None, f"FTS table {table} should exist"
    
    def test_migration_creates_triggers(self, db_instance):
        """Test that FTS triggers are created."""
        with db_instance.transaction() as cursor:
            # Check for some key triggers
            triggers = [
                "topics_ai", "topics_ad", "topics_au",
                "flashcards_ai", "flashcards_ad", "flashcards_au"
            ]

            for trigger in triggers:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND name=?",
                    (trigger,)
                )
                result = cursor.fetchone()
                assert result is not None, f"Trigger {trigger} should exist"
    
    def test_schema_version_updated(self, db_instance):
        """Test that schema version is correctly updated to the current code version."""
        with db_instance.transaction() as cursor:
            cursor.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = 'rag_char_chat_schema'"
            )
            result = cursor.fetchone()
            assert result is not None
            assert result["version"] == CharactersRAGDB._CURRENT_SCHEMA_VERSION
