The text below explains how the EHIS dataset metadata file is structured and the ideas behind certain choices.
Similar ways are used to handle other datasets used in this package although actual implementations may vary.
All attempts were made to make this approach as consistent as possible.

# Top level
Top level keys in the provided YAML file have been introduced for grouping purposes only;
there is some nesting present meaning these groups can have subgroups etc.
Each leaf in this tree structure is a column in the dataset.

## Individual column metadata
Each column must have at least its description, type, and acceptable values. 
All other entries are optional and only appear when the feature behaviour deviates from the default.

```yaml
ACTUAL_COLUMN_NAME:
  Description: DESCRIPTION STRING
  Type: id | indicator | ordinal | category | weight | date
  Name: lower_case_string_no_spaces
  Target: person | household | strata | sampling_unit
  DataType:
  Answer categories and codes:
  Filter:
  Anonymisation rule:
```

The actual order of keys is irrelevant although following the same pattern makes the scheme more readable.

# Type
## IDs
The id `DataType` defaults to `uint64[pyarrow]` unless specified otherwise.

 - Personal IDs are at least non-negative to clearly differentiate between individuals
 - Other columns can have a smaller range and contain negative values to indicate missing or otherwise unapplicable data

## Indicators
Any column that's comprised of two unique entries only is seen as an indicator.
The mapping to `bool[pyarrow]` is always the same across the dataset i.e. 

```python
{1: True, 2: False}
```

However, exceptions are acceptable at later stages of data processing 
when a categorical or an ordinal variable has been reduced to two unique values only.

This is done mostly for space-saving purposes.

## Dates
These are still being worked on.

## Weights
At least non-negative `float32[pyarrow]` values.

## Categories & ordinals
By default, all values are seen as `int8[pyarrow]`.

# Target
The default target is `person` as most records are of individual character.
