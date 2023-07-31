import abc

class EnvironmentWrapper(abc.ABC):
    '''
    EnvironmentWrapper define
    the interface between real environment and agent wanted environment.
    '''

    def step(self, action, action_settings = None):
        ''' Do an action to environment and return the 
        next observation, reward, is done(terminal), and info_dict from environment'''
        raise NotImplementedError("This function step is not implemented yet.")

    def reset(self, index=-1, reset_settings = None):
        ''' reset the environment'''
        raise NotImplementedError("This function reset is not implemented yet.")

    def get_action_space(self, index=0):
        raise NotImplementedError("This function get_action_space is not implemented yet.")

    def get_agent_count(self):
        raise NotImplementedError("This function get_agent_count is not implemented yet.")

    def need_restart(self):
        return False
