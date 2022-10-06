from enum import IntFlag


class LDT_PATTERN(IntFlag):
    NOTOUR = 0
    BEGIN = 1  # leave home, do not return home today
    END = 2  # returns home today from overnight trip
    COMPLETE = 3  # long distance day-trip
    AWAY = 4  # long dist tour middle of multiday


# note that COMPLETE == BEGIN|END, this is used for finding tours that need trips


LDT_PATTERN_BITSHIFT = 3
