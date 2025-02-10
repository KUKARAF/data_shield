from typing import Dict, Type
import importlib
import pkgutil
import nltk
from pathlib import Path
from anonymizer.base import BaseFilter

def ensure_nltk_resources():
    """
    Download all required NLTK resources.
    """
    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]

    # First verify and download core resources
    for resource in resources:
        try:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt'
                              else f'taggers/{resource}' if 'tagger' in resource
                              else f'chunkers/{resource}' if 'chunker' in resource
                              else f'corpora/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Failed to download NLTK resource {resource}: {e}")
            raise

    # Additional verification for critical resources
    critical_resources = {
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
        'tokenizers/punkt': 'punkt',
        'chunkers/maxent_ne_chunker': 'maxent_ne_chunker',
        'corpora/words': 'words'
    }

    for path, resource in critical_resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Critical resource {resource} not found. Attempting download...")
            nltk.download(resource, quiet=True)
            # Verify download
            try:
                nltk.data.find(path)
            except LookupError as e:
                raise RuntimeError(f"Failed to download critical resource {resource}") from e

def load_filters() -> Dict[str, BaseFilter]:
    """
    Dynamically load all available filters from the filters directory.

    Returns:
        Dictionary mapping filter names to filter instances
    """
    filters = {}
    filters_dir = Path(__file__).parent / "filters"

    # Ensure NLTK resources are available
    ensure_nltk_resources()

    for module_info in pkgutil.iter_modules([str(filters_dir)]):
        if module_info.name.startswith('__'):
            continue

        module = importlib.import_module(f"anonymizer.filters.{module_info.name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseFilter) and
                attr != BaseFilter):
                filter_name = module_info.name
                filters[filter_name] = attr()

    return filters