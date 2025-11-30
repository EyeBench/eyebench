"""__init__.py"""

from beartype import BeartypeConf  # <-- this isn't your fault
from beartype.claw import beartype_this_package  # <-- boilerplate for victory

beartype_this_package(
    conf=BeartypeConf(violation_type=UserWarning)
)  # <-- yay! your team just won
