"""Property name mapping for developability properties."""

# Primary properties: Simplified names to dataset column names
PRIMARY_PROPERTIES = {
    "hydrophobicity": "HIC",
    "self-association": "AC-SINS_pH7.4",
    "ac-sins": "AC-SINS_pH7.4",  # Fallback for AC-SINS
    "titer": "Titer",
    "thermostability": "Tm2",
    "polyreactivity": "PR_CHO",
}

# Reverse mapping for display
COLUMN_TO_NAME = {v: k for k, v in PRIMARY_PROPERTIES.items()}

# Additional properties
ADDITIONAL_PROPERTIES = {
    "purity": "Purity",
    "sec-monomer": "SEC_%Monomer",
    "tm1": "Tm1",
    "hac": "HAC",
    "tonset": "Tonset",
    "ac-sins-ph6": "AC-SINS_pH6.0",
}


def get_dataset_column(property_name: str) -> str:
    """
    Convert simplified property name to dataset column name.

    Args:
        property_name: Simplified property name (e.g., 'hydrophobicity', 'HIC')

    Returns:
        Dataset column name (e.g., 'HIC', 'AC-SINS_pH7.4')

    Raises:
        ValueError: If property name is not recognized
    """
    # Already a dataset column name
    if property_name in PRIMARY_PROPERTIES.values():
        return property_name

    if property_name in ADDITIONAL_PROPERTIES.values():
        print(f"⚠️  WARNING: '{property_name}' is not a primary property.")
        print(f"    Primary properties are: {', '.join(PRIMARY_PROPERTIES.keys())}")
        return property_name

    # Convert from simplified name
    lower_name = property_name.lower()

    if lower_name in PRIMARY_PROPERTIES:
        return PRIMARY_PROPERTIES[lower_name]

    if lower_name in ADDITIONAL_PROPERTIES:
        print(f"⚠️  WARNING: '{property_name}' is not a primary property.")
        print(f"    Primary properties are: {', '.join(PRIMARY_PROPERTIES.keys())}")
        return ADDITIONAL_PROPERTIES[lower_name]

    # If not found, raise error with helpful message
    raise ValueError(
        f"Unknown property: '{property_name}'\n"
        f"Primary properties: {', '.join(PRIMARY_PROPERTIES.keys())}\n"
        f"Additional properties: {', '.join(ADDITIONAL_PROPERTIES.keys())}\n"
        f"Or use dataset column names directly: {', '.join(PRIMARY_PROPERTIES.values())}"
    )


def get_display_name(column_name: str) -> str:
    """
    Convert dataset column name to simplified display name.

    Args:
        column_name: Dataset column name (e.g., 'HIC', 'AC-SINS_pH7.4')

    Returns:
        Simplified display name (e.g., 'hydrophobicity', 'self-association')
    """
    if column_name in COLUMN_TO_NAME:
        return COLUMN_TO_NAME[column_name]

    # Check additional properties
    for name, col in ADDITIONAL_PROPERTIES.items():
        if col == column_name:
            return name

    return column_name


def is_primary_property(property_name: str) -> bool:
    """
    Check if a property is a primary developability target.

    Args:
        property_name: Property name (simplified or column name)

    Returns:
        True if primary property, False otherwise
    """
    lower_name = property_name.lower()
    return (
        lower_name in PRIMARY_PROPERTIES or
        property_name in PRIMARY_PROPERTIES.values()
    )



def list_primary_properties() -> list:
    """Return list of primary property names (simplified)."""
    return list(PRIMARY_PROPERTIES.keys())


def list_all_properties() -> dict:
    """Return dict of all available properties (simplified: column)."""
    return {**PRIMARY_PROPERTIES, **ADDITIONAL_PROPERTIES}
