import pytest
from anonymizer.filters.name import NameFilter
from anonymizer.filters.id import IdFilter

def test_name_filter():
    """Test name detection with various forms and honorifics."""
    filter = NameFilter()

    text = """
    Dr. Jane Wilson
    Pan Tadeusz
    Mr. John Smith and Professor Alice Johnson are collaborating.
    """
    names = filter.find(text)

    assert any(n['full'] == 'Dr. Jane Wilson' for n in names)
    assert any(n['title'] == 'Dr.' and n['first'] == 'Jane' and n['last'] == 'Wilson' for n in names)
    assert any(n['full'] == 'Pan Tadeusz' for n in names)
    assert any(n['full'] == 'Mr. John Smith' for n in names)
    assert any(n['full'] == 'Professor Alice Johnson' for n in names)

def test_name_filter_components():
    """Test name component detection."""
    filter = NameFilter()

    text = "Dr. John Smith Jr."
    names = filter.find(text)

    assert len(names) == 1
    name = names[0]
    assert name['title'] == 'Dr.'
    assert name['first'] == 'John'
    assert name['last'] == 'Smith Jr.'

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