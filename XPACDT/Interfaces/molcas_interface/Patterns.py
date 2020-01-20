import re

RAW_PATTERNS = dict()

# Match the version of MOLCAS as written in the .log file
# Capture the version string.
RAW_PATTERNS["molcas version"] = r"""
    [Vv]ersion
    \s*             # Spaces
    (               # Start of the version capturing group
        [-v\.\d]*   # Version string
    )"""

RAW_PATTERNS["error"] = r"""
    (                       # Start capture group for the section
        ---\ Start\ Module
        .*                  # The whole content of the section
        ---\ Stop\ Module
        .*
        rc=
        \s*
            (.*)            # The error type
        \s*
        ---
    )
    """

# Match decimal numbers in the format used in MOLCAS .log file
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
# Capture the gradient vector
# Note: the vector captured by this pattern can be directly parse by
# np.fromstring(captured_vector, sep=" ")
RAW_PATTERNS["gradient"] = r"""
    \w+             # Atom symbol
    \d+             # ID (integer)
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

# Match the content of an overlap matrix section
# Capture all the matrix elements at once
# If the first group is not empty, that means the matrix is given by its
# diagonal only
# Note: the vector representing the diagonal matrix captured by this pattern
# can be directly parse by np.fromstring(captured_vector, sep=" ")
RAW_PATTERNS["overlap matrix"] = r"""
    (               # Start capturing group for diagonal text
        Diagonal,\ with\ elements\n
        \ *
    )?              # The group is optional (match empty string if absent)
    ({num}|\s)*     # Capture diagonal vector
    """.format(num=RAW_PATTERNS["decimal"])

# Compiled patterns
PATTERNS = {key: re.compile(pattern, re.DOTALL | re.VERBOSE)
            for key, pattern in RAW_PATTERNS.values()}
