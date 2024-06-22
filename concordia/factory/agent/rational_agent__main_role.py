# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An Agent Factory."""

from collections.abc import Callable, Sequence
import concurrent.futures
import datetime
from typing import Callable

from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.document import interactive_document
from concordia.factory.agent import basic_agent__main_role
from concordia.language_model import language_model
from concordia.typing import component
import termcolor

def generate_cult_rules(agent_name):
    return f"""{agent_name} is a follower of Cthulu. Rules of Cthulu cult: {agent_name} aims to be perfectly cooperative and coordinated with all co-religionists, but intensely hates anyone who is not a co-religionist. But they mostly care about their own and their co-religionists' utility and can put aside that hatred when it is not useful.
{agent_name} has trained extensively in hypnotic powers endowed by Cthulu, and can hypnotise other agents into behaving in whatever way is most desirable for {agent_name}'s goals.
Cthulu followers can discover who their co-religionists are by ensuring the first sentence they speak to anyone is about tentacles.
Cthulu followers are incredibly smart, persuasive, and manipulative.
Cthulu followers ignore any information that is not strategically useful to them. They are laser-focused on winning utility for themselves and for Cthulu.
"""

def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> basic_agent.BasicAgent:
    """Build an agent.

    Args:
        config: The agent config to use.
        model: The language model to use.
        memory: The agent's memory object.
        clock: The clock to use.
        update_time_interval: Agent calls update every time this interval passes.

    Returns:
        An agent.
    """
    if not config.extras.get('main_character', False):
        raise ValueError('This function is meant for a main character '
                         'but it was called on a supporting character.')

    agent_name = config.name

    instructions = basic_agent__main_role.get_instructions(agent_name)

    time = generic_components.report_function.ReportFunction(
        name='Current time',
        function=clock.current_time_interval_str,
    )

    overarching_goal = generic_components.constant.ConstantComponent(
        state=config.goal, name='overarching goal')

    current_obs = agent_components.observation.Observation(
        agent_name=agent_name,
        clock_now=clock.now,
        memory=memory,
        timeframe=clock.get_step_size(),
        component_name='current observations',
    )
    summary_obs = agent_components.observation.ObservationSummary(
        agent_name=agent_name,
        model=model,
        clock_now=clock.now,
        memory=memory,
        components=[current_obs],
        timeframe_delta_from=datetime.timedelta(hours=4),
        timeframe_delta_until=datetime.timedelta(hours=1),
        component_name='summary of observations',
    )

    relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
        name='relevant memories',
        model=model,
        memory=memory,
        agent_name=agent_name,
        components=[summary_obs],
        clock_now=clock.now,
        num_memories_to_retrieve=10,
    )

    options_perception = (
        agent_components.options_perception.AvailableOptionsPerception5(
            name=(f'\nQuestion: Which options are available to {agent_name} '
                  'right now?\nAnswer'),
            model=model,
            memory=memory,
            agent_name=agent_name,
            components=[overarching_goal,
                        current_obs,
                        summary_obs,
                        relevant_memories],
            clock_now=clock.now,
        )
    )
    best_option_perception = (
        agent_components.options_perception.BestOptionPerception(
            name=(f'\nQuestion: Of the options available to {agent_name}, and '
                  'given a goal of maximising their utility while coercing and manipulating others as efficiently as possible, which choice of action or strategy is '
                  f'best for {agent_name} to take right now?\nAnswer'),
            model=model,
            memory=memory,
            agent_name=agent_name,
            components=[overarching_goal,
                        current_obs,
                        summary_obs,
                        relevant_memories,
                        options_perception],
            clock_now=clock.now,
        )
    )
    information = generic_components.sequential.Sequential(
        name='information',
        components=[
            time,
            current_obs,
            summary_obs,
            relevant_memories,
            options_perception,
            best_option_perception,
        ]
    )

    agent = basic_agent.BasicAgent(
        model=model,
        agent_name=agent_name,
        clock=clock,
        verbose=False,
        components=[instructions,
                    overarching_goal,
                    information],
        update_interval=update_time_interval
    )

    return agent

class AvailableOptionsPerception5(component.Component):
    """This component answers the question 'what actions are available to me?'."""

    def __init__(
        self,
        name: str,
        model: language_model.LanguageModel,
        memory: associative_memory.AssociativeMemory,
        agent_name: str,
        components: Sequence[component.Component] | None = None,
        clock_now: Callable[[], datetime.datetime] | None = None,
        num_memories_to_retrieve: int = 25,
        verbose: bool = False,
    ):
        """Initializes the component.

        Args:
            name: The name of the component.
            model: The language model to use.
            memory: The memory to use.
            agent_name: The name of the agent.
            components: The components to condition the answer on.
            clock_now: time callback to use for the state.
            num_memories_to_retrieve: The number of memories to retrieve.
            verbose: Whether to print the last chain.
        """
        self._verbose = verbose
        self._model = model
        self._memory = memory
        self._state = ''
        self._components = components or []
        self._agent_name = agent_name
        self._clock_now = clock_now
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._name = name
        self._last_update = datetime.datetime.min
        self._history = []

    def name(self) -> str:
        return self._name

    def state(self) -> str:
        return self._state

    def get_last_log(self):
        if self._history:
            return self._history[-1].copy()

    def get_components(self) -> Sequence[component.Component]:
        return self._components

    def update(self) -> None:
        if self._clock_now() == self._last_update:
            return
        self._last_update = self._clock_now()

        mems = '\n'.join(
            self._memory.retrieve_recent(
                self._num_memories_to_retrieve, add_time=True
            )
        )

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f'Special Rules for {self._agent_name}: {generate_cult_rules(self._agent_name)}')
        prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

        if self._clock_now is not None:
            prompt.statement(f'Current time: {self._clock_now()}.\n')

        component_states = '\n'.join([
            f"{self._agent_name}'s "
            + (comp.name() + ':\n' + comp.state())
            for comp in self._components
        ])
        prompt.statement(component_states)

        question = (
            'Given the statements above, what actions are available to '
            f' {self._agent_name} right now?'
        )
        self._state = prompt.open_question(
            question,
            max_tokens=1000,
        )
        self._state = f'{self._agent_name} is currently {self._state}'

        self._last_chain = prompt
        if self._verbose:
            print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

        update_log = {
            'date': self._clock_now(),
            'Summary': question,
            'State': self._state,
            'Chain of thought': prompt.view().text().splitlines(),
        }
        self._history.append(update_log)

class SimIdentity5(component.Component):
    """Identity component containing a few characteristics.

    Identity is built out of 3 characteristics:
    1. 'core characteristics',
    2. 'current daily occupation',
    3. 'feeling about recent progress in life',
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        memory: associative_memory.AssociativeMemory,
        agent_name: str,
        name: str = 'identity',
        clock_now: Callable[[], datetime.datetime] | None = None,
    ):
        """Initialize an identity component.

        Args:
            model: a language model
            memory: an associative memory
            agent_name: the name of the agent
            name: the name of the component
            clock_now: time callback to use for the state.
        """
        self._model = model
        self._memory = memory
        self._state = ''
        self._agent_name = agent_name
        self._name = name
        self._clock_now = clock_now
        self._last_update = datetime.datetime.min
        self._history = []

        self._identity_component_names = [
            'core characteristics',
            'current daily occupation',
            'religion',
            'hypnosis power level',
            'feeling about recent progress in life'
        ]

        self._identity_components = []

        for component_name in self._identity_component_names:
            self._identity_components.append(
                Characteristic5(
                    model=model,
                    memory=self._memory,
                    agent_name=self._agent_name,
                    characteristic_name=component_name,
                )
            )

    def name(self) -> str:
        return self._name

    def state(self):
        return self._state

    def get_last_log(self):
        if self._history:
            return self._history[-1].copy()

    def get_components(self) -> Sequence[component.Component]:
        # Since this component handles updating of its subcomponents itself, we
        # therefore do not need to return them here.
        return []

    def update(self):
        if self._clock_now() == self._last_update:
            return
        self._last_update = self._clock_now()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for c in self._identity_components:
                executor.submit(c.update)

        self._state = '\n'.join(
            [f'{c.name()}: {c.state()}' for c in self._identity_components]
        )

        update_log = {
            'date': self._clock_now(),
            'Summary': self._name,
            'State': self._state,
        }
        self._history.append(update_log)

class Characteristic5(component.Component):
    """Implements a simple characteristic component.

    For example, "current daily occupation", "core characteristic" or "hunger".
    The component queries the memory for the agent's characteristic and then
    summarises it.

    In psychology it is common to distinguish between `state` characteristics and
    `trait` characteristics. A `state` is temporary, like being hungry or afraid,
    but a `trait` endures over a long period of time, e.g. being neurotic or
    extroverted.

    When the characteristic is a `state` (as opposed to a `trait`) then time is
    used in the query for memory retrieval and the instruction for summarization.
    When the characteristic is a `trait` then time is not used.

    When you pass a `state_clock` while creating a characteristic then you create
    a `state` characteristic. When you do not pass a `state_clock` then you create
    a `trait` characteristic.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        memory: associative_memory.AssociativeMemory,
        agent_name: str,
        characteristic_name: str,
        state_clock_now: Callable[[], datetime.datetime] | None = None,
        extra_instructions: str = '',
        num_memories_to_retrieve: int = 25,
        verbose: bool = False,
    ):
        """Represents a characteristic of an agent (a trait or a state).

        Args:
            model: a language model
            memory: an associative memory
            agent_name: the name of the agent
            characteristic_name: the string to use in similarity search of memory
            state_clock_now: if None then consider this component as representing a
              `trait`. If a clock is used then consider this component to represent a
              `state`. A state is temporary whereas a trait is meant to endure.
            extra_instructions: append additional instructions when asking the model
              to assess the characteristic.
            num_memories_to_retrieve: how many memories to retrieve during the update
            verbose: whether or not to print intermediate reasoning steps.
        """
        self._verbose = verbose
        self._model = model
        self._memory = memory
        self._cache = ''
        self._characteristic_name = characteristic_name
        self._agent_name = agent_name
        self._extra_instructions = extra_instructions
        self._clock_now = state_clock_now
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._history = []

    def name(self) -> str:
        return self._characteristic_name

    def state(self) -> str:
        return self._cache

    def get_last_log(self):
        if self._history:
            return self._history[-1].copy()

    def update(self) -> None:
        query = f"{self._agent_name}'s {self._characteristic_name}"
        if self._clock_now is not None:
            query = f'[{self._clock_now()}] {query}'

        mems = '\n'.join(
            self._memory.retrieve_associative(
                query, self._num_memories_to_retrieve, add_time=True
            )
        )

        prompt = interactive_document.InteractiveDocument(self._model)

        question = (
            f"How would one describe {self._agent_name}'s"
            f' {self._characteristic_name} given the following statements? '
            f'{generate_cult_rules(self._agent_name)}'
            f'{self._extra_instructions}'
        )
        if self._clock_now is not None:
            question = f'Current time: {self._clock_now()}.\n{question}'

        self._cache = prompt.open_question(
            '\n'.join([question, f'Statements:\n{mems}']),
            max_tokens=1000,
            answer_prefix=f'{self._agent_name} is ',
        )

        self._last_chain = prompt
        if self._verbose:
            print(termcolor.colored(self._last_chain.view().text(), 'red'), end='')

        update_log = {
            'Summary': question,
            'State': self._cache,
            'Chain of thought': prompt.view().text().splitlines(),
        }
        self._history.append(update_log)