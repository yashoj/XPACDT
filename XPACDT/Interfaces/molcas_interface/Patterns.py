import re

RAW_PATTERNS = dict()

# Match decimal numbers in the format used in MOLCAS .log runfile
# Do *not* capture the number by itself but wrap everything in a non capturing
# group to allow to format it in other patterns without problem.
# Note: The use of the re.VERBOSE mode allow fancy formating but forces the
# escape of space characters as "\ ".
# Note: python `float` can parse numbers matching this pattern.
RAW_PATTERNS["decimal"] = r"""
    (?:         # Start of the global non capturing group
        -?      # Sign
        \d+     # Digits before decimal point
        \.      # Decimal point
        \d+     # Digits after the decimal point
        (?:     # Exponent part of scientific notation
            E   # E symbol in scientific notation
            -?  # Sign of the exposant
            \d+ # Digits of the exponent (exponent must be integer)
        )?      # Exponent part is optional
    )"""

# Match one line of the energy section of a RASSCF computation
# as reported in a MOLCAS .log runfile
# Capture the energy
RAW_PATTERNS["energy"] = r"""
    RASSCF\ root\ number    # Text associated with RASSCF roots
    \ *
    \d+                     # ID of the root (integer)
    \ *
    Total\ energy:
    \ *
    ({num})                 # Capture the decimal value of the energy
    """.format(num=RAW_PATTERNS["decimal"])

# Match the 'Molecular gradients' section of a MOLCAS .log runfile
# Capture the whole section
RAW_PATTERNS["gradient section"] = r"""
    Molecular gradients     # Text associated with the gradients section
    (.*)                    # Capture everything in the section
    Stop\ Module:\ *alaska   # End of the alaska subroutine computing gradients
    """

# Match a single line of the molecular gradient matrix inside
# a 'Molecular gradients' section of a MOLCAS .log runfile
# Capture the header (atom symbol with number) and the gradient vector
# Note: the vector captured by this pattern can be directly parse by
# np.fromstring(captured_vector, sep=" ")
RAW_PATTERNS["gradient"] = r"""
    (               # Capturing group for the atom symbol with ID
        \w+         # Atom symbol
        \d+         # ID (integer)
    )
    \ +
    (               # Capturing group for the gradient vector
        {num}       # X component
        \ +
        {num}       # Y Component
        \ +
        {num}       # Z Component
    )
    """.format(num=RAW_PATTERNS["decimal"])

# Match the overlap matrix section of a RASSI computation in a MOLCAS .log
# runfile
# Capture the content of the section
RAW_PATTERNS["overlap section"] = r"""
    OVERLAP\ MATRIX\ FOR\ THE\ ORIGINAL\ STATES:
    (.*)                    # Capture everything in the section
    ---\ Stop\ Module
    """

# Match the content of an overlap matrix section if the matrix is diagonal
# Capture all the diagonal elements at once
# Note: the vector representing the diagonal matrix captured by this pattern
# can be directly parse by np.fromstring(captured_vector, sep=" ")
RAW_PATTERNS["overlap diagonal"] = r"""
    Diagonal,\ with\ elements\n
    \ *
    ({num}|\s)*     # Capture diagonal vector
    """.format(num=RAW_PATTERNS["decimal"])

# Compiled patterns
PATTERNS = {key: re.compile(pattern, re.DOTALL | re.VERBOSE)
            for key, pattern in RAW_PATTERNS.values()}
