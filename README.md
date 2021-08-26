The files which should be downloaded and run are frazil_plume_del_rho.py and simplified_frazil_plume.py.

frazil_plume_del_rho.py is the main file for the project.
(It is called this to differentiate it from the previous attempt at coding the frazil plume equations, frazil_plume_equations.py, in T and S, which hasn't been debugged.)
The only things which should need changing when running it are the boolean values (my_precipitation, Jenkins_ambient, vary_radius, simplified_plume, and linear_ice),
the ambient conditions, and the very end of the code which plots different aspects.
Ideally I should separate out the methods for making it run and the code for actually running it into at least two different files, but I haven't gotten around to that yet.
There's also the code for the height as a function of the distance along the slope, and associated with that for sin(theta) which has not yet been done,
so this code will need to be substantially and functionally changed.

The other file which is still being studied at this point is simplified_frazil_plume.py.
This one attempts to implement the simplified system of only 2 differential equations, but the results it outputs do not yet agree with those in the main plume file,
so it might also have to be substantially changed.

Most other files were used for testing something rather than being the main experiment -
radius_and_velocity.py is the most recent of these, which looks at the radius as a function of velocity (and radius as a function of plume width) for different drag types,
while stokes_and_jenkins_drags.py, done slightly earlier, looked at which of the hypothesized drag forces dominated.
magorrian_wells_odes.py and magorrian_wells_with_pressure.py have the original versions of the code, replicating the Magorrian & Wells paper,
which was duplicated and adapted to include frazil ice.
get_M_without_abc.py was an alternative function for calculating M, at a point when the code was producing error messages for the normal method,
and dimensionless_magorrian_odes.py was meant to be an alternative version of the initial magorrian_wells_odes.py file at a time I was struggling with that file.
