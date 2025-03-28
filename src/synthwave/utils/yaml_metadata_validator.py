from typing import ClassVar

class YAMLMetadataValidator:

    DEFAULT_ALLOWED_LEAF_ENTRIES: ClassVar[list[str, ...]] = ["Description", "Type", "Target", "Name", "DataType", "Answer categories and codes", "Filter", "Anonymisation rule"]
    allowed_leaf_entries: ClassVar[list[str, ...]] = DEFAULT_ALLOWED_LEAF_ENTRIES

    DEFAULT_MINIMAL_LEAF_ENTRIES: ClassVar[set[str, ...]] = {"Description", "Type", "Answer categories and codes"}
    minimal_leaf_entries: ClassVar[set[str, ...]] = DEFAULT_MINIMAL_LEAF_ENTRIES

    DEFAULT_ALLOWED_TARGETS: ClassVar[list[str, ...]] = ["person", "household", "strata", "sampling_unit"]
    allowed_targets: ClassVar[list[str, ...]] = DEFAULT_ALLOWED_TARGETS

    DEFAULT_ALLOWED_TYPES = [None, 'category', 'date', 'id', 'indicator', 'ordinal', 'weight']
    allowed_types: ClassVar[list[str | None, ...]] = DEFAULT_ALLOWED_TYPES
    #TODO must not contain incomplete records

    ALLOWED_DATA_TYPES = []

    @staticmethod
    def build_name(_leaf: dict) -> str:
        _name_parts = [_leaf["Type"], _leaf["Target"] if "Target" in _leaf.keys() else "person"]

        if "Name" in _leaf.keys():
            _name_parts.append(_leaf["Name"])

        return '_'.join(_name_parts)

    @staticmethod
    def is_valid_name(_name) -> None:
        if (" " in _name) or (not _name.islower()):
            raise NameError(f"Invalid name: {_name}")

    @classmethod
    def update_config(cls, config_data: dict) -> None:
        """Update class variables."""

        if 'allowed_leaf_entries' in config_data:
            cls.allowed_leaf_entries = config_data['allowed_leaf_entries']

        if 'minimal_leaf_entries' in config_data:
            cls.minimal_leaf_entries = config_data['minimal_leaf_entries']

        if 'allowed_targets' in config_data:
            cls.allowed_targets = config_data['allowed_targets']

        if 'allowed_types' in config_data:
            cls.allowed_types = config_data['allowed_types']


    @classmethod
    def reset_config(cls) -> None:
        """Reset configuration to defaults."""
        cls.allowed_leaf_entries = cls.DEFAULT_ALLOWED_LEAF_ENTRIES
        cls.minimal_leaf_entries = cls.DEFAULT_MINIMAL_LEAF_ENTRIES
        cls.allowed_targets = cls.DEFAULT_ALLOWED_TARGETS
        cls.allowed_types = cls.DEFAULT_ALLOWED_TYPES

    @classmethod
    def check_targets(cls, _targets: list[str, ...]) -> None:
        if not set(_targets).issubset(cls.allowed_targets):
            raise ValueError(f"Invalid target present: {set(_targets).difference(cls.allowed_targets)}")

    @classmethod
    def check_types(cls, _types: list[str, ...]) -> None:
        if not set(_types).issubset(cls.allowed_types):
            raise ValueError(f"Invalid type present: {set(_types).difference(cls.allowed_types)}")

    @classmethod
    def check_duplicate_keys(cls, _metadata: dict) -> None:
        """Check if duplicate keys are present."""
        _keys = []
        for _v in cls.extract_columns(_metadata):
            _keys += list(_v.keys())

        if len(_keys) != len(set(_keys)):
            seen = set()
            duplicates = []

            for _k in _keys:
                if _k in seen:
                    duplicates.append(_k)
                else:
                    seen.add(_k)
            raise KeyError(f"Duplicate keys: {duplicates}")

    @classmethod
    def is_minimal_leaf(cls, _leaf: dict) -> bool:
        return cls.minimal_leaf_entries.issubset(_leaf.keys())

    @classmethod
    def is_pure_leaf(cls, _leaf: dict) -> bool:
        return set(_leaf.keys()).issubset(cls.allowed_leaf_entries)

    @classmethod
    def extract_columns(cls, d: dict):
        # to be a leaf it must be minimal and pure
        for k, v in d.items():

            if v is None:
                raise ValueError(f"Empty value {k, v} in the tree")

            if isinstance(v, dict):
                if cls.is_minimal_leaf(v):
                    if cls.is_pure_leaf(v):
                        yield {k: v}
                    else:
                        raise ValueError(f"Metadata is malformed, leaf {k, v} is contaminated, check {v.keys()}")
                else:
                    yield from cls.extract_columns(v)
            else:
                raise ValueError(f"Metadata is malformed, tree terminates with an atypical leaf {k, v}")
