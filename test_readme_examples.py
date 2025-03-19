from anonymizer import Anonymizer

def main():
    # Basic usage example
    print('\nTesting basic usage example:')
    anonymizer = Anonymizer()
    text = 'Hi John Smith, your ID is ABC-123.'
    anonymized = anonymizer.hide_personal_data(text)
    print('Original:', text)
    print('Anonymized:', anonymized)
    restored = anonymizer.fill_personal_data(anonymized)
    print('Restored:', restored)

    # Email anonymization example
    print('\nTesting email anonymization:')
    anonymizer = Anonymizer(filters=['name', 'id'])
    email = '''
    Dear Dr. Jane Wilson,

    Regarding your account number 12345-ABC, we have processed your request.
    Best regards,
    Robert Johnson
    Technical Support
    '''
    anonymized = anonymizer.hide_personal_data(email)
    print('Original:', email)
    print('Anonymized:', anonymized)
    restored = anonymizer.fill_personal_data(anonymized)
    print('Restored:', restored)

if __name__ == '__main__':
    main()
