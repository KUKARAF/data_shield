import pytest
from anonymizer.filters.name import NameFilter
from anonymizer.filters.date import DateFilter
from anonymizer.filters.id import IdFilter

def test_name_filter():
    """Test name detection with various forms and honorifics."""
    filter = NameFilter()

    text = """
    Drogi Panie Tadeuszu,
    Pan Tadeusz jest ważny.
    Mr. John Smith and Professor Alice Johnson are collaborating.
    """
    names = filter.find(text)

    assert "Panie Tadeuszu" in names
    assert "Pan Tadeusz" in names
    assert "Mr. John Smith" in names
    assert "Professor Alice Johnson" in names

def test_date_filter():
    """Test date detection."""
    filter = DateFilter()

    text = """Dates: 
        2023/12/25
        December 25th, 2023
        12-25-2023
    """
    dates = filter.find(text)

    assert len(dates) == 3
    assert any("December 25th, 2023" in d for d in dates)
    assert any("2023/12/25" in d for d in dates)

def test_id_filter():
    """Test ID detection."""
    filter = IdFilter()

    text = """IDs:
        case 12345
        Reference ABC-123-XYZ
        ID number 123456
    """
    ids = filter.find(text)

    assert len(ids) == 3
    assert "12345" in ids
    assert "ABC-123-XYZ" in ids
    assert "123456" in ids