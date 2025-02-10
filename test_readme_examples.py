from anonymizer import Anonymizer

def main():
    # Basic usage example
    print('\nTesting basic usage example:')
    anonymizer = Anonymizer()
    text = 'Dear John Smith, your case number ABC-123 has been processed.'
    anonymized = anonymizer.hide_personal_data(text)
    print('Original:', text)
    print('Anonymized:', anonymized)
    restored = anonymizer.fill_personal_data(anonymized)
    print('Restored:', restored)

    # Testing specific filters
    print('\nTesting specific filters:')
    anonymizer = Anonymizer(filters=['name'])
    text = 'Dear Dr. Jane Wilson, your reference ID is XYZ-789.'
    anonymized = anonymizer.hide_personal_data(text)
    print('Original:', text)
    print('Anonymized:', anonymized)

    # Testing grammar preservation
    print('\nTesting grammar preservation:')
    anonymizer = Anonymizer(preserve_grammar=True)
    text = 'A John Smith is an important person.'
    anonymized = anonymizer.hide_personal_data(text)
    print('Original:', text)
    print('Anonymized:', anonymized)

    # Testing LLM integration example
    print('\nTesting LLM integration example:')
    name_remover = Anonymizer(filters=['name'])
    text = 'Hi John, how was your week?'
    anonymized = name_remover.hide_personal_data(text)
    print('Original:', text)
    print('Anonymized:', anonymized)

    # Simulate LLM response
    transformed_text = "Dear <NAME_1>, I hope this email finds you well and your week has been pleasant."
    print('LLM Response:', transformed_text)

    final_text = name_remover.fill_personal_data(transformed_text)
    print('Restored:', final_text)

if __name__ == '__main__':
    main()