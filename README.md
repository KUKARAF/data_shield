pip install text-anonymizer
```

## Basic Usage

```python
from anonymizer import Anonymizer

# Initialize with default filters (name and id)
anonymizer = Anonymizer()

# Anonymize text
text = "Dear John Smith, your case number ABC-123 has been processed."
anonymized = anonymizer.hide_personal_data(text)
print(anonymized)
# Output: "Dear <NAME_1>, your case number <ID_1> has been processed."

# Restore original text
restored = anonymizer.fill_personal_data(anonymized)
print(restored)
# Output: "Dear John Smith, your case number ABC-123 has been processed."
```

## Advanced Usage

### Specific Filters

```python
# Initialize with specific filters
anonymizer = Anonymizer(filters=['name'])  # Only anonymize names
text = "Dear Dr. Jane Wilson, your reference ID is XYZ-789."
anonymized = anonymizer.hide_personal_data(text)
print(anonymized)
# Output: "Dear Dr. <NAME_1>, your reference ID is XYZ-789."
```

### Grammar Preservation

The anonymizer maintains grammatical correctness by adjusting articles ("a"/"an") based on the pronunciation of placeholders:

```python
anonymizer = Anonymizer(preserve_grammar=True)
text = "A John Smith is an important person."
anonymized = anonymizer.hide_personal_data(text)
print(anonymized)
# Output: "A <NAME_1> is an important person."
```

## Real-World Examples

### Medical Records

```python
# Anonymizing medical records
medical_record = """
Patient: Sarah Johnson
ID: MED-2024-567
Referring Doctor: Dr. Michael Brown
Notes: Patient was referred by Dr. Brown on January 15th.
"""

anonymizer = Anonymizer()
anonymized_record = anonymizer.hide_personal_data(medical_record)
print(anonymized_record)
# Output:
"""
Patient: <NAME_1>
ID: <ID_1>
Referring Doctor: Dr. <NAME_2>
Notes: Patient was referred by Dr. <NAME_2> on January 15th.
"""
```

### Legal Documents

```python
# Anonymizing legal documents
legal_doc = """
AGREEMENT between John A. Smith (ID: ABC123) and 
Mary Wilson (ID: DEF456), represented by 
Atty. James Thompson.
"""

anonymizer = Anonymizer()
anonymized_doc = anonymizer.hide_personal_data(legal_doc)
print(anonymized_doc)
# Output:
"""
AGREEMENT between <NAME_1> (ID: <ID_1>) and 
<NAME_2> (ID: <ID_2>), represented by 
Atty. <NAME_3>.
"""
```

### LLM Integration Example

```python
# Using the anonymizer with an LLM to rephrase text while preserving privacy
name_remover = Anonymizer(filters=['name'])  # Only anonymize names
original_text = "Hi John, how was your week?"

# Anonymize names before sending to LLM
anonymized = name_remover.hide_personal_data(original_text)
print(anonymized)
# Output: "Hi <NAME_1>, how was your week?"

# Send to LLM (example using a hypothetical LLM function)
def send_to_llm(text):
    # Your LLM API call here
    return "Dear <NAME_1>, I hope this email finds you well and your week has been pleasant."

transformed_text = send_to_llm(anonymized)
print(transformed_text)
# Output: "Dear <NAME_1>, I hope this email finds you well and your week has been pleasant."

# Restore the original names
final_text = name_remover.fill_personal_data(transformed_text)
print(final_text)
# Output: "Dear John, I hope this email finds you well and your week has been pleasant."
```

## Extending with Custom Filters

You can create custom filters by extending the `BaseFilter` class:

```python
from anonymizer import BaseFilter
from typing import List

class CustomFilter(BaseFilter):
    def find(self, text: str) -> List[str]:
        # Implement your pattern matching logic here
        pass

# Use your custom filter
anonymizer = Anonymizer(filters=['name', 'id', 'custom'])