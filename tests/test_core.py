import pytest
from anonymizer.core import Anonymizer

def test_initialization():
    """Test Anonymizer initialization with different parameters."""
    # Default initialization
    anon = Anonymizer()
    assert len(anon._filters) > 0

    # Specific filters
    anon = Anonymizer(filters=['name', 'id'])
    assert len(anon._filters) == 2
    assert 'name' in anon._filters
    assert 'id' in anon._filters

def test_hide_personal_data():
    """Test personal data anonymization."""
    anon = Anonymizer()

    # Test with various data types
    text = "Dear Dr. John Doe, your case 12345 is being processed"
    anonymized = anon.hide_personal_data(text)

    assert "Dear Dr. <FIRST_NAME_1> <LAST_NAME_1>" in anonymized
    assert "<ID_1>" in anonymized
    assert "John Doe" not in anonymized
    assert "12345" not in anonymized

def test_name_with_title():
    """Test name anonymization with titles."""
    anon = Anonymizer(filters=['name'])

    text = "Dr. Jane Wilson wrote to Professor Smith"
    anonymized = anon.hide_personal_data(text)

    assert "Dr. <FIRST_NAME_1> <LAST_NAME_1>" in anonymized
    assert "Professor <FIRST_NAME_2>" in anonymized
    assert "Jane Wilson" not in anonymized
    assert "Smith" not in anonymized

def test_fill_personal_data():
    """Test restoration of personal data."""
    anon = Anonymizer()

    original = "Dear Dr. John Doe, your case 12345 is important"
    anonymized = anon.hide_personal_data(original)
    restored = anon.fill_personal_data(anonymized)

    assert restored == original

def test_empty_input():
    """Test handling of empty input."""
    anon = Anonymizer()

    assert anon.hide_personal_data("") == ""
    assert anon.fill_personal_data("") == ""

def test_grammar_preservation():
    """Test grammatical correctness preservation."""
    anon = Anonymizer(preserve_grammar=True)

    text = "I am a John Smith"
    anonymized = anon.hide_personal_data(text)

    assert "a <FIRST_NAME_1> <LAST_NAME_1>" in anonymized

def test_multiple_names():
    """Test handling of multiple names in text."""
    anon = Anonymizer(filters=['name'])

    text = "Dr. Jane Wilson and Mr. Robert Brown had a meeting"
    anonymized = anon.hide_personal_data(text)

    assert "Dr. <FIRST_NAME_1> <LAST_NAME_1>" in anonymized
    assert "Mr. <FIRST_NAME_2> <LAST_NAME_2>" in anonymized