import inspect

from activitysim.core.exceptions import WhaleAccessError

NO_DEFAULT = "< no default >"


class WhaleAccessor:
    """
    Boilerplate code for accessors.

    Accessors consolidate groups of related functions in a common interface,
    without requiring the main Whale class to become bloated by including all
    relevant functionality.  They also allow setting and storing attributes
    without worrying about conflicting with similarly named attributes of
    other accessors.
    """

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, objtype=None):
        from .state import Whale

        assert isinstance(instance, Whale)
        self.obj = instance
        return self

    def __set__(self, instance, value):
        raise ValueError(f"cannot directly set accessor {self._name}")

    def __delete__(self, instance):
        raise ValueError(f"cannot delete accessor {self._name}")


class FromWhale:
    def __init__(self, member_type=None, default_init=False, default_value=NO_DEFAULT):
        """
        Creates a property to access an element from the current context.

        Parameters
        ----------
        member_type : type
            The data type for this attribute.  Basic type validation may be
            applied when setting this value, but validation could be disabled
            when optimizing models for production and this type checking should
            not be relied on as a runtime feature. If not given, member_type is
            read from the accessor's type annotation (if applicable).
        default_init : bool
            When set to true, if this context value is accessed and it has not
            already been set, it is automatically initialized with the default
            value (i.e. via a no-argument constructor) for the given type.
        default_value : Any, optional
            When set to some value, if this context value is accessed and it has
            not already been set, it is automatically initialized with the this
            default value.

        """
        self.member_type = member_type
        self._default_init = default_init
        self._default_value = default_value
        if self._default_init and self._default_value != NO_DEFAULT:
            raise ValueError("cannot use both default_init and default_value")

    def __set_name__(self, owner, name):
        self.name = f"{owner.__name__.lower()}_{name}"
        # set member type based on annotation
        if self.member_type is None:
            annot = inspect.get_annotations(owner)
            if name in annot:
                self.member_type = annot[name]

    def __get__(self, instance: WhaleAccessor, objtype=None):
        try:
            return instance.obj.context[self.name]
        except (KeyError, AttributeError):
            if self._default_init:
                instance.obj.context[self.name] = self.member_type()
                return instance.obj.context[self.name]
            elif self._default_value != NO_DEFAULT:
                instance.obj.context[self.name] = self._default_value
                return instance.obj.context[self.name]
            raise WhaleAccessError(f"{self.name} not initialized for this whale")

    def __set__(self, instance: WhaleAccessor, value):
        if not self.__validate_type(value):
            raise TypeError(f"{self.name} must be {self.member_type} not {type(value)}")
        instance.obj.context[self.name] = value

    def __delete__(self, instance):
        self.__set__(instance, None)

    def __validate_type(self, value):
        # type validation is only done at the top level for now.
        try:
            type_ok = isinstance(value, self.member_type)
        except:
            from typing import get_args, get_origin

            type_ok = isinstance(value, get_origin(self.member_type))
        return type_ok