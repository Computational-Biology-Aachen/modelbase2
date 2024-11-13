import math  # noqa: F401  # models might need it
import re
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import libsbml
import numpy as np  # noqa: F401  # models might need it

from modelbase2.model import Model
from modelbase2.sbml._data import (
    AtomicUnit,
    Compartment,
    CompositeUnit,
    Compound,
    Derived,
    Function,
    Parameter,
    Reaction,
)
from modelbase2.sbml._mathml import parse_sbml_math
from modelbase2.sbml._name_conversion import _name_to_py
from modelbase2.sbml._unit_conversion import get_operator_mappings, get_unit_conversion

UNIT_CONVERSION = get_unit_conversion()
OPERATOR_MAPPINGS = get_operator_mappings()
INDENT = "    "


@dataclass(slots=True)
class Parser:
    # Collections
    boundary_species: set[str] = field(default_factory=set)
    # Parsed stuff
    atomic_units: dict[str, AtomicUnit] = field(default_factory=dict)
    composite_units: dict[str, CompositeUnit] = field(default_factory=dict)
    compartments: dict[str, Compartment] = field(default_factory=dict)
    parameters: dict[str, Parameter] = field(default_factory=dict)
    variables: dict[str, Compound] = field(default_factory=dict)
    derived: dict[str, Derived] = field(default_factory=dict)
    initial_assignment: dict[str, Derived] = field(default_factory=dict)
    functions: dict[str, Function] = field(default_factory=dict)
    reactions: dict[str, Reaction] = field(default_factory=dict)

    def parse(self, file: str | Path) -> Self:
        if not Path(file).exists():
            msg = "Model file does not exist"
            raise OSError(msg)
        doc = libsbml.readSBMLFromFile(str(file))

        # Check for unsupported packages
        for i in range(doc.num_plugins):
            if doc.getPlugin(i).getPackageName() == "comp":
                msg = "No support for comp package"
                warnings.warn(msg, stacklevel=1)

        sbml_model = doc.getModel()
        if sbml_model is None:
            return self

        if bool(sbml_model.getConversionFactor()):
            msg = "Conversion factors are currently not supported"
            warnings.warn(msg, stacklevel=1)

        self.parse_functions(sbml_model)
        self.parse_units(sbml_model)
        self.parse_compartments(sbml_model)
        self.parse_variables(sbml_model)
        self.parse_parameters(sbml_model)
        self.parse_initial_assignments(sbml_model)
        self.parse_rules(sbml_model)
        self.parse_constraints(sbml_model)
        self.parse_reactions(sbml_model)
        self.parse_events(sbml_model)
        return self

    ###############################################################################
    # PARSING STAGE
    ###############################################################################

    def parse_constraints(self, sbml_model: libsbml.Model) -> None:
        if len(sbml_model.getListOfConstraints()) > 0:
            warnings.warn(
                "modelbase does not support model constraints. "
                "Check the file for hints of the range of stable solutions.",
                stacklevel=1,
            )

    def parse_events(self, sbml_model: libsbml.Model) -> None:
        if len(sbml_model.getListOfEvents()) > 0:
            warnings.warn(
                "modelbase does not current support events. "
                "Check the file for how to integrate properly.",
                stacklevel=1,
            )

    def parse_units(self, sbml_model: libsbml.Model) -> None:
        for unit_definition in sbml_model.getListOfUnitDefinitions():
            composite_id = unit_definition.getId()
            local_units = []
            for unit in unit_definition.getListOfUnits():
                atomic_unit = AtomicUnit(
                    kind=UNIT_CONVERSION[unit.getKind()],
                    scale=unit.getScale(),
                    exponent=unit.getExponent(),
                    multiplier=unit.getMultiplier(),
                )
                local_units.append(atomic_unit.kind)
                self.atomic_units[atomic_unit.kind] = atomic_unit
            self.composite_units[composite_id] = CompositeUnit(
                sbml_id=composite_id,
                units=local_units,
            )

    def parse_compartments(self, sbml_model: libsbml.Model) -> None:
        for compartment in sbml_model.getListOfCompartments():
            sbml_id = _name_to_py(compartment.getId())
            size = compartment.getSize()
            if str(size) == "nan":
                size = 0
            self.compartments[sbml_id] = Compartment(
                name=compartment.getName(),
                dimensions=compartment.getSpatialDimensions(),
                size=size,
                units=compartment.getUnits(),
                is_constant=compartment.getConstant(),
            )

    def parse_parameters(self, sbml_model: libsbml.Model) -> None:
        for parameter in sbml_model.getListOfParameters():
            self.parameters[_name_to_py(parameter.getId())] = Parameter(
                value=parameter.getValue(),
                is_constant=parameter.getConstant(),
            )

    def parse_variables(self, sbml_model: libsbml.Model) -> None:
        for compound in sbml_model.getListOfSpecies():
            compound_id = _name_to_py(compound.getId())
            if bool(compound.getConversionFactor()):
                warnings.warn(
                    "modelbase does not support conversion factors. "
                    f"Pleas check {compound_id} manually",
                    stacklevel=1,
                )

            # NOTE: What the shit is this?
            initial_amount = compound.getInitialAmount()
            if str(initial_amount) == "nan":
                initial_amount = compound.getInitialConcentration()
                is_concentration = str(initial_amount) != "nan"
            else:
                is_concentration = False

            has_boundary_condition = compound.getBoundaryCondition()
            if has_boundary_condition:
                self.boundary_species.add(compound_id)

            self.variables[compound_id] = Compound(
                compartment=compound.getCompartment(),
                initial_amount=initial_amount,
                substance_units=compound.getSubstanceUnits(),
                has_only_substance_units=compound.getHasOnlySubstanceUnits(),
                has_boundary_condition=has_boundary_condition,
                is_constant=compound.getConstant(),
                is_concentration=is_concentration,
            )

    def parse_functions(self, sbml_model: libsbml.Model) -> None:
        for func in sbml_model.getListOfFunctionDefinitions():
            func_name = func.getName()
            sbml_id = func.getId()
            if sbml_id is None or sbml_id == "":
                sbml_id = func_name
            elif func_name is None or func_name == "":
                func_name = sbml_id
            func_name = _name_to_py(func_name)

            if (node := func.getMath()) is None:
                continue
            body, args = parse_sbml_math(node=node)

            self.functions[func_name] = Function(
                body=body,
                args=args,
            )

    ###############################################################################
    # Different kinds of derived values
    ###############################################################################

    def parse_initial_assignments(self, sbml_model: libsbml.Model) -> None:
        for assignment in sbml_model.getListOfInitialAssignments():
            name = _name_to_py(assignment.getSymbol())
            node = assignment.getMath()
            if node is None:
                warnings.warn(
                    f"Unusable math for {name}",
                    stacklevel=1,
                )
                continue

            body, args = parse_sbml_math(node)
            self.initial_assignment[name] = Derived(
                body=body,
                args=args,
            )

    def _parse_algebraic_rule(self, rule: libsbml.AlgebraicRule) -> None:
        warnings.warn(f"Skipping algebraic rule rule {rule.getId()}", stacklevel=1)

    def _parse_assignment_rule(self, rule: libsbml.AssignmentRule) -> None:
        if (node := rule.getMath()) is None:
            return

        name: str = _name_to_py(rule.getId())
        body, args = parse_sbml_math(node=node)

        self.derived[name] = Derived(
            body=body,
            args=args,
        )

    def _parse_rate_rule(self, rule: libsbml.RateRule) -> None:
        warnings.warn(f"Skipping rate rule {rule.getId()}", stacklevel=1)

    def parse_rules(self, sbml_model: libsbml.Model) -> None:
        """Parse rules and separate them by type"""
        for rule in sbml_model.getListOfRules():
            if rule.element_name == "algebraicRule":
                self._parse_algebraic_rule(rule=rule)
            elif rule.element_name == "assignmentRule":
                self._parse_assignment_rule(rule=rule)
            elif rule.element_name == "rateRule":
                self._parse_rate_rule(rule=rule)
            else:
                msg = "Unknown rate type"
                raise ValueError(msg)

    def _parse_local_parameters(
        self, reaction_id: str, kinetic_law: libsbml.KineticLaw
    ) -> dict[str, str]:
        """Parse local parameters"""
        parameters_to_update = {}
        for parameter in kinetic_law.getListOfLocalParameters():
            old_id = _name_to_py(parameter.getId())
            if old_id in self.parameters:
                new_id = f"{reaction_id}__{old_id}"
                parameters_to_update[old_id] = new_id
            else:
                new_id = old_id
            self.parameters[new_id] = Parameter(
                value=parameter.getValue(),
                is_constant=parameter.getConstant(),
            )
        # Some models apparently also write local parameters in this
        for parameter in kinetic_law.getListOfParameters():
            old_id = _name_to_py(parameter.getId())
            if old_id in self.parameters:
                new_id = f"{reaction_id}__{old_id}"
                parameters_to_update[old_id] = new_id
            else:
                new_id = old_id
            self.parameters[new_id] = Parameter(
                value=parameter.getValue(),
                is_constant=parameter.getConstant(),
            )
        return parameters_to_update

    def parse_reactions(self, sbml_model: libsbml.Model) -> None:
        for reaction in sbml_model.getListOfReactions():
            sbml_id = _name_to_py(reaction.getId())
            kinetic_law = reaction.getKineticLaw()
            if kinetic_law is None:
                continue
            parameters_to_update = self._parse_local_parameters(
                reaction_id=sbml_id,
                kinetic_law=kinetic_law,
            )

            node = reaction.getKineticLaw().getMath()
            body, args = parse_sbml_math(node=node)

            # Update parameter references
            for old, new in parameters_to_update.items():
                pat = re.compile(f"({old})" + r"\b")
                body = pat.sub(new, body)

            for i, arg in enumerate(args):
                args[i] = parameters_to_update.get(arg, arg)

            parsed_reactants: defaultdict[str, int] = defaultdict(int)
            for substrate in reaction.getListOfReactants():
                species = _name_to_py(substrate.getSpecies())
                if species not in self.boundary_species:
                    stoichiometry = substrate.getStoichiometry()
                    if str(stoichiometry) == "nan":
                        msg = f"Cannot parse stoichiometry: {stoichiometry}"
                        raise ValueError(msg)
                    parsed_reactants[species] -= stoichiometry
            parsed_products: defaultdict[str, int] = defaultdict(int)
            for product in reaction.getListOfProducts():
                species = _name_to_py(product.getSpecies())
                if species not in self.boundary_species:
                    stoichiometry = product.getStoichiometry()
                    if str(stoichiometry) == "nan":
                        msg = f"Cannot parse stoichiometry: {stoichiometry}"
                        raise ValueError(msg)
                    parsed_products[species] += stoichiometry

            self.reactions[sbml_id] = Reaction(
                body=body,
                stoichiometry=dict(parsed_reactants) | dict(parsed_products),
                args=args,
            )


def _handle_fn(name: str, body: str, args: list[str]) -> Callable[..., float]:
    func_args = ", ".join(args)
    func_str = "\n".join(
        [
            f"def {name}({func_args}):",
            f"{INDENT}return {body}",
            "",
        ]
    )
    try:
        exec(func_str, globals(), None)  # noqa: S102
    except SyntaxError as e:
        msg = f"Invalid function definition: {func_str}"
        raise SyntaxError(msg) from e
    python_func = globals()[name]
    python_func.__source__ = func_str
    return python_func  # type: ignore


def _translate(sbml: Parser) -> Model:
    m = Model()

    # Parameters
    for k, v in sbml.parameters.items():
        if k in sbml.derived:
            continue
        m.add_parameter(k, v.value)

    # Variables
    for k, v in sbml.variables.items():
        if k in sbml.derived:
            continue
        m.add_variable(k, v.initial_amount)

    # Compartments
    for k, v in sbml.compartments.items():
        if k in sbml.derived:
            continue
        if v.is_constant:
            m.add_parameter(k, v.size)
        else:
            m.add_variable(k, v.size)

    # Derived
    for k, v in sbml.derived.items():
        m.add_derived(
            k,
            fn=_handle_fn(k, body=v.body, args=v.args),
            args=v.args,
            sort_derived=False,
        )

    # Globally create functions. Yes, this sucks
    for k, v in sbml.functions.items():
        _handle_fn(k, body=v.body, args=v.args)

    # Calculate initial assignments
    # FIXME: probably will need to sort these ...
    for k, v in sbml.initial_assignment.items():
        args = m._get_args(m._variables, include_readouts=False)  # noqa: SLF001
        fn = _handle_fn(k, body=v.body, args=v.args)
        m.update_variable(k, fn(*(args[arg] for arg in v.args)))

    # Reactions
    for k, v in sbml.reactions.items():
        m.add_reaction(
            name=k,
            fn=_handle_fn(k, body=v.body, args=v.args),
            stoichiometry=v.stoichiometry,
            args=v.args,
        )

    return m


def from_sbml(file: Path | str) -> Model:
    return _translate(Parser().parse(file=file))