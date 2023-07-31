import abc

class ObservationPreprocessor(abc.ABC):
	'''
	ObservationPreprocessor define 
	the prcoess of raw observation output from environment to agent needed.
	'''
	def __init__(self, config):
		self.observation_dimension = config["observation_dimension"]

	@abc.abstractmethod
	def process(self, observation, process_settings = None):
		''' 
		Return the processed observation for agent using.
		Provide extra process settings parameter for other extensions
		'''
		raise NotImplementedError("This function process is not implemented yet.")
