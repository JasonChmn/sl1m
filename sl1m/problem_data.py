class ProblemData:
    """
    Class to store the result of the planning problem
    """

    def __init__(self, success, time, coms=None, moving_feet_pos=None, all_feet_pos=None, surface_indices=None,
                 comb_needed=None, comb_total=None, list_surfaces=None, obj_value=None):
        self.success = success
        self.time = time
        self.coms = coms
        self.moving_feet_pos = moving_feet_pos
        self.all_feet_pos = all_feet_pos
        self.surface_indices = surface_indices
        # Added
        self.comb_needed = comb_needed
        self.comb_total = comb_total
        self.list_surfaces = list_surfaces
        self.obj_value = obj_value

    def __str__(self):
        string = "ProblemData: "
        string += "\n \t Success: " + str(self.success)
        string += "\n \t Time: " + str(self.time)
        return string

    def __repr__(self):
        return self.__str__()
