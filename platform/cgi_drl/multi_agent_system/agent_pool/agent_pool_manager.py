# coding=utf-8
import os
from collections import namedtuple, deque, defaultdict
import numpy as np

AgentMeta = namedtuple("AgentMeta", ["full_name", "series_name", "model_directory", "rating"])
class AgentPoolManager:
    from_agent_pool_mark = "_#from_agent_pool"
    shared_agent_mark = "_#shared"

    @staticmethod
    def is_from_agent_pool(name):
        """
        Return:
            Wheather the name of agent is sampled from the agent pool 
        """
        return name.endswith(AgentPoolManager.from_agent_pool_mark)
    
    @staticmethod
    def is_shared(name):
        """
        Return:
            Wheather the name of agent is shared from the existing agent 
        """
        return name.endswith(AgentPoolManager.shared_agent_mark)

    @staticmethod
    def remove_from_agent_pool_mark(name):
        return (
            name[: -len(AgentPoolManager.from_agent_pool_mark)]
            if AgentPoolManager.is_from_agent_pool(name)
            else name
        )

    def __init__(self, base_directory, config):
        self._base_directory = base_directory + "/agent_pool"
        if not os.path.exists(self._base_directory):
            os.makedirs(self._base_directory)

        self._pool_capacity = config["pool_capacity"]
        self._use_score_rating = config["use_score_rating"]
        self._default_rating = config[
            "default_score_rating" if self._use_score_rating else "default_elo_rating"
        ]

        # For rating (e.g., Elo rating)
        self._k_factor = config["k_factor"]

        # Name -> AgentMeta
        self._agent_pool = {}
        # Keep the name of last n(pool_size) agent
        # agent_type -> deque of `agent_type`
        self._agent_name_db = defaultdict(lambda: deque(maxlen=self._pool_capacity))
        # Keep the name of static agent (will not be remove), usually the training one.
        # And also not count in the capacity (i.e. not in the self.agent_name_db)
        self._static_agent_db = set()
        self._rng = np.random.default_rng()

        for agent_info in config["existing_agents"]:
            self.load_existing_agent(agent_info["full_name"], agent_info["series_name"], agent_info["model_directory"])

    def _remove_agent_mapping(self, name):
        """Will only remove the key,value in the self._agent_pool"""
        del self._agent_pool[name]

    def get_pool_size(self, series_name="_all"):
        """Will not include the static agent."""
        if series_name == "_all":
            return len(self._agent_pool)
        return len(self._agent_name_db[series_name])

    def add_agent_to_pool(self, full_name, series_name, save_model_callback, **kwargs):
        """
        Args:
            kwargs:
                static: the agent will be put into the static set.
        """
        if full_name in self._agent_pool:
            raise RuntimeError(f"Agent pool already has name {full_name}.")

        model_directory = os.path.join(self._base_directory, full_name)
        if os.path.exists(model_directory):
            pass
        else:
            os.makedirs(model_directory)
        # Save the model
        save_model_callback(model_directory)
        rating = kwargs.get("rating", self._default_rating)
        is_static = kwargs.get("is_static", False)
        if is_static:
            self._static_agent_db.add(full_name)
        else:
            agent_names_deque = self._agent_name_db[series_name]
            # Keep n last model
            if len(agent_names_deque) == agent_names_deque.maxlen:
                # Remove the oldest agent
                self._remove_agent_mapping(agent_names_deque[0])
            agent_names_deque.append(full_name)

        # Add in the pool
        self._agent_pool[full_name] = AgentMeta(full_name, series_name, model_directory, rating)

    def load_existing_agent(self, full_name, series_name, model_directory):
        print(full_name, series_name, model_directory)
        if full_name in self._agent_pool:
            raise RuntimeError(f"Agent pool already has name {full_name}.")

        if not os.path.exists(model_directory):
            raise RuntimeError(f"model_directory: {model_directory}, not existing.")
        
        self._static_agent_db.add(full_name)
        self._agent_pool[full_name] = AgentMeta(full_name, series_name, model_directory, self._default_rating)
        print(f"Load Agent: {full_name} as {series_name} from {model_directory}")


    def sample_agent_with_prioritized_fictitious_self_play(self, anchor_agent_name, series_name) -> AgentMeta:
        """Use anchor agent to calculate the expected score (e.g. winnrate)
        of each agent.
        Then weighted by the function and sample from `series_name`.
        """
        assert (not self._use_score_rating), "Prioritized sampling does not support score rating"
        # Sample the agent that around target's own rating
        def pfsp_f(x):
            return x * (1 - x)

        def get_expected_scores(anchor_elo, candidates_elo):
            return 1.0 / (1.0 + np.power(10, (candidates_elo - anchor_elo) / 400))

        _pool_size = self.get_pool_size(series_name)
        if _pool_size <= 0:
            raise RuntimeError("Pool should not be empty.")

        anchor = self._agent_pool[anchor_agent_name]
        candidates = self._agent_name_db[series_name]

        expected_scores = get_expected_scores(
            anchor.rating, np.array([self._agent_pool[c].rating for c in candidates])
        )
        weighted_scores = pfsp_f(expected_scores)
        prob = weighted_scores / np.sum(weighted_scores)
        name = candidates[self._rng.choice(len(candidates), p=prob)]
        return self._agent_pool[name]

    def sample_agent_from_pool_uniformly(self, series_name) -> AgentMeta:
        """Will not sample the agent in the static set"""
        _pool_size = self.get_pool_size(series_name)
        if _pool_size <= 0:
            raise RuntimeError("Pool should not be empty.")
        # Uniformly
        name = self._agent_name_db[series_name][self._rng.integers(0, _pool_size)]
        return self._agent_pool[name]

    def get_agent_meta(self, full_name) -> AgentMeta:
        if full_name not in self._agent_pool:
            raise RuntimeError(f"Agent({full_name}) did not exist.")
        return self._agent_pool[full_name]

    def update_elo_rating(self, score_dict, use_softmax_score=False):
        """
        Args:
            score_dict: Dict[str, float],: (name, score)
                e.g. ("alice_solver", average winrate)
        """
        def softmax(x):
            # Normalize
            x -= np.max(x)
            # Softmax
            exp = np.exp(x)
            p = exp / np.sum(exp)
            return p

        def get_expected_scores(x):
            return 1.0 / (1.0 + np.power(10, (x[::-1] - x) / 400))

        for name in score_dict.keys():
            if name not in self._agent_pool:
                raise RuntimeError(f"Agent({name}) did not exist.")
        # Not update if two names are the same.
        name = list(score_dict.keys())
        if len(name) == 1:
            return

        actual_scores = np.array([score for score in score_dict.values()])
        if use_softmax_score:
            actual_scores = softmax(actual_scores)

        elo_ratings = np.array(
            [self._agent_pool[name].rating for name in score_dict.keys()]
        )
        expected_scores = get_expected_scores(elo_ratings)
        new_rating = elo_ratings + self._k_factor * (actual_scores - expected_scores)
        diff_rating = new_rating - elo_ratings

        update_result = {}
        for name, r, diff_r in zip(score_dict.keys(), new_rating, diff_rating):
            self._agent_pool[name] = self._agent_pool[name]._replace(rating=r)
            update_result[name] = [r, diff_r]

        return update_result

    def update_score_rating(self, score_dict):
        """
        Args:
            score_dict: Dict[str, float],: (name, score)
                e.g. ("alice_solver", average winrate)
        """
        for name in score_dict.keys():
            if name not in self._agent_pool:
                raise RuntimeError(f"Agent({name}) did not exist.")
        # Not update if two names are the same.
        name = list(score_dict.keys())
        if name[0] == name[1]:
            return
        actual_scores = np.array([score for score in score_dict.values()])

        old_ratings = np.array([self._agent_pool[name].rating for name in score_dict.keys()])
        new_rating = old_ratings + self._k_factor * (actual_scores - old_ratings)

        update_result = {}
        for name, r, diff_r in zip(score_dict.keys(), new_rating, diff_rating):
            self._agent_pool[name] = self._agent_pool[name]._replace(rating=r)
            update_result[name] = [r, diff_r]

        return update_result
