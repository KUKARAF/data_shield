# Text Anonymizer

A Python package for text anonymization using NLTK, with support for names, IDs, and email content.

## Installation

```bash
pip install text-anonymizer
```

## Quick Start

```python
from anonymizer import Anonymizer

# Initialize anonymizer
anonymizer = Anonymizer()

# Basic text anonymization
text = "Hi John Smith, your ID is ABC-123."
anonymized = anonymizer.hide_personal_data(text)
print(anonymized)
# Output: "Hi <FIRST_NAME_1> <LAST_NAME_1>, your ID is <ID_1>."

# Restore original text
restored = anonymizer.fill_personal_data(anonymized)
print(restored)
# Output: "Hi John Smith, your ID is ABC-123."
```

## Email Anonymization Example

```python
# Initialize anonymizer with name and ID filters
anonymizer = Anonymizer(filters=['name', 'id'])

# Anonymize email content
email = """
Dear Dr. Jane Wilson,

Regarding your account number 12345-ABC, we have processed your request.
Best regards,
Robert Johnson
Technical Support
"""

anonymized = anonymizer.hide_personal_data(email)
print(anonymized)
# Output:
"""
Dear Dr. <FIRST_NAME_1> <LAST_NAME_1>,

Regarding your account number <ID_1>, we have processed your request.
Best regards,
<FIRST_NAME_2> <LAST_NAME_2>
Technical Support
"""
```

## Features

- Name detection with title preservation (Dr., Mr., Prof., etc.)
- ID number anonymization
- Email content processing
- Original text restoration
- Grammatically correct anonymization

## License

MIT