from fenics import *
from dolfin import *
from mshr import *
from petsc4py import PETSc
from slepc4py import SLEPc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import subprocess, os, shlex, shutil


def move_vertex_to_circle(mesh, vertex, radius):
    """
    Moves vertices to interface and boundary circles, returns the new mesh
    """
    if not near(vertex.point().norm(), radius):
        phi = np.arctan2(vertex.x(1), vertex.x(0))
        mesh.coordinates()[vertex.index()] = [radius * cos(phi), radius * sin(phi)]
    return mesh


def refine_and_adapt(mesh, boundaries, domains, radius, delta):
    """
    Refines mesh and moves vertices to interface and boundary circles,
    generates new MeshFunction with interface and boundary data
    """
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    mesh_refined = refine(mesh, MeshFunction("bool", mesh, mesh.topology().dim(), True))

    domains_new = adapt(domains, mesh_refined)
    boundaries_new = adapt(boundaries, mesh_refined)

    for facet in facets(mesh_refined):
        if boundaries_new[facet] == 1:
            for vertex in vertices(facet):
                mesh_refined = move_vertex_to_circle(mesh_refined, vertex, radius)
        elif boundaries_new[facet] == 2:
            for vertex in vertices(facet):
                mesh_refined = move_vertex_to_circle(
                    mesh_refined, vertex, radius + delta
                )
    return mesh_refined, boundaries_new, domains_new


def adjust_mesh(mesh, sub_domains, radius, delta):
    """
    Moves vertices to interface and boundary circles
    """
    inner_mesh = SubMesh(mesh, sub_domains, 1)
    interface = BoundaryMesh(inner_mesh, "exterior")
    vertex_map_B2S = interface.entity_map(0)
    vertex_map_S2M = inner_mesh.data().array("parent_vertex_indices", 0)
    for vertex in vertices(interface):
        mesh = move_vertex_to_circle(
            mesh, Vertex(mesh, vertex_map_S2M[vertex_map_B2S[vertex.index()]]), radius
        )

    b_mesh = BoundaryMesh(mesh, "exterior")
    for vertex in vertices(b_mesh):
        mesh = move_vertex_to_circle(
            mesh, Vertex(mesh, b_mesh.entity_map(0)[vertex.index()]), radius + delta
        )
    return mesh


def test_run(ver, iters, radius, ratio, phys_params, t_final, num_steps, tol):
    """
    Base run for the linear parabolic system
    All parameters are set to unity except for
    radius = 10; delta  = 5;
    ratio := delta/radius = 0.5

    physics parameters:
        d3:     diffusion coefficient for TG
        d2:     diffusion coefficient for DG
        mu:     reaction rate for TG
        nu:     reaction rate for DG
        kappa:  feedback rate in the reservoir region; DG --> TG
        v_m:    maximum reaction rate for TG hydrolysis (from Michaelis-Menten)
        k_m:    Michaelis constant
        ctotal: total concentrations in the whole droplet
    """
    lim = 1e9  # system size limit

    # System Parameters
    delta = ratio * radius

    d3 = Constant(phys_params["d3"])
    d2 = Constant(phys_params["d2"])
    mu = Constant(phys_params["mu"])
    nu = Constant(phys_params["nu"])
    v_m = Constant(phys_params["v_m"])
    k_m = Constant(phys_params["k_m"])
    kappa = Constant(phys_params["kappa"])
    ctotal = Constant(phys_params["ctotal"])

    # Time Parameters
    dt = t_final / num_steps
    tau = Constant(dt)

    # Generate Mesh
    print("\n1 Generating Geometry and Mesh...")
    mesh = None
    h_data = [2 ** (i + 2) for i in range(0, iters)]
    for i in range(0, iters):
        h = 2 ** (-(i + 2))
        print("Mesh resolution: {} in {}".format(int(h ** (-1)), h_data))
        droplet = Circle(Point(0, 0), radius + delta)
        reservoir = Circle(Point(0, 0), radius)
        active = droplet - reservoir
        droplet.set_subdomain(1, reservoir)
        droplet.set_subdomain(2, active)

        if mesh is None:
            mesh = generate_mesh(droplet, h ** (-1))
            boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
            domains = MeshFunction(
                "size_t", mesh, mesh.topology().dim(), mesh.domains()
            )
            # Mesh adjustment, optional
            mesh = adjust_mesh(mesh, domains, radius, delta)
            mesh_size = mesh.hmin()
            for facet in facets(mesh):
                if near(
                    sum([vert.point().norm() for vert in vertices(facet)]),
                    mesh.topology().dim() * radius,
                    eps=1e-3 * mesh_size,
                ):
                    boundaries[facet] = 1
                elif near(
                    sum([vert.point().norm() for vert in vertices(facet)]),
                    mesh.topology().dim() * (radius + delta),
                    eps=1e-3 * mesh_size,
                ):
                    boundaries[facet] = 2
        else:
            mesh, boundaries, domains = refine_and_adapt(
                mesh, boundaries, domains, radius, delta
            )

        mesh_1 = SubMesh(mesh, domains, 1)
        mesh_2 = SubMesh(mesh, domains, 2)

        dx = Measure("dx", domain=mesh, subdomain_data=domains)
        dS = Measure("dS", domain=mesh, subdomain_data=boundaries, subdomain_id=1)

        P1 = FiniteElement("P", triangle, 1)
        V = FunctionSpace(mesh, MixedElement([P1, P1, P1, P1]))
        V_inner = FunctionSpace(mesh_1, "CG", 1)
        V_outer = FunctionSpace(mesh_2, "CG", 1)

        # Define break condition
        if V_inner.dim() * 2 + V_outer.dim() * 2 >= lim:
            print("System size restriction reached:", str(lim))
            break
        mesh.bounding_box_tree().build(mesh)

    print("\n2 Defining spaces and variational form...")
    (
        p_1,
        p_2,
        p_3,
        p_4,
    ) = TestFunctions(V)
    u, v, w, z = TrialFunctions(V)

    # Initial data, piecewise positive constants
    c_0 = Expression(("0.5", "0.5", "0.5", "0.5"), degree=1)
    c_0 = interpolate(c_0, V)
    u_1, v_1, w_1, z_1 = c_0.split()

    u_1 = interpolate(u_1, V_inner)
    u_1.set_allow_extrapolation(True)
    v_1 = interpolate(v_1, V_inner)
    v_1.set_allow_extrapolation(True)
    w_1 = interpolate(w_1, V_outer)
    w_1.set_allow_extrapolation(True)
    z_1 = interpolate(z_1, V_outer)
    z_1.set_allow_extrapolation(True)

    total_mass_previous = assemble((u_1 + v_1) * dx(1) + (w_1 + z_1) * dx(2))

    # Save the FEM solutions every time step
    vtkfile_u = File("run{}/paraview/mesh{}/fem_u.pvd".format(ver, int(h ** (-1))))
    vtkfile_v = File("run{}/paraview/mesh{}/fem_v.pvd".format(ver, int(h ** (-1))))
    vtkfile_w = File("run{}/paraview/mesh{}/fem_w.pvd".format(ver, int(h ** (-1))))
    vtkfile_z = File("run{}/paraview/mesh{}/fem_z.pvd".format(ver, int(h ** (-1))))

    vtkfile_u << (u_1, 0)
    vtkfile_v << (v_1, 0)
    vtkfile_w << (w_1, 0)
    vtkfile_z << (z_1, 0)

    # Bilinear Form
    F = (
        (u / tau) * p_1 * dx(1)
        + dot(d3 * grad(u), grad(p_1)) * dx(1)
        - kappa * v * p_1 * dx(1)
        - mu * (w("+") - u("-")) * p_1("-") * dS(1)
        + (v / tau) * p_2 * dx(1)
        + dot(d2 * grad(v), grad(p_2)) * dx(1)
        + kappa * v * p_2 * dx(1)
        - nu * (z("+") - v("-")) * p_2("-") * dS(1)
        + (w / tau) * p_3 * dx(2)
        + dot(d3 * grad(w), grad(p_3)) * dx(2)
        + (v_m / (k_m + w_1)) * w * p_3 * dx(2)
        + mu * (w("+") - u("-")) * p_3("+") * dS(1)
        + (z / tau) * p_4 * dx(2)
        + dot(d2 * grad(z), grad(p_4)) * dx(2)
        - (v_m / (k_m + w_1)) * w * p_4 * dx(2)
        + nu * (z("+") - v("-")) * p_4("+") * dS(1)
    )

    # Linear Form
    B = (
        (u_1 / tau) * p_1 * dx(1)
        + (v_1 / tau) * p_2 * dx(1)
        + (w_1 / tau) * p_3 * dx(2)
        + (z_1 / tau) * p_4 * dx(2)
    )

    u_l2_err = []
    v_l2_err = []
    w_l2_err = []
    z_l2_err = []
    time = []

    A = assemble(F)
    PETScOptions.set("mat_mumps_icntl_24", 1)
    solver = PETScLUSolver(as_backend_type(A), "mumps")
    sol = Function(V)

    print("\n3 Solving the nonlinear system...")

    progress = Progress("Time-stepping", num_steps)
    t = 0
    for n in range(1, num_steps + 1):
        t += dt
        b = assemble(B)
        solver.solve(sol.vector(), b)

        # Save solution to file
        u, v, w, z = sol.split(deepcopy=True)

        # Update previous solution
        u = interpolate(u, V_inner)
        u.set_allow_extrapolation(True)
        v = interpolate(v, V_inner)
        v.set_allow_extrapolation(True)
        w = interpolate(w, V_outer)
        w.set_allow_extrapolation(True)
        z = interpolate(z, V_outer)
        z.set_allow_extrapolation(True)

        u_l2_err_ = assemble((u - u_1) ** 2 * dx(1)) ** 0.5
        v_l2_err_ = assemble((v - v_1) ** 2 * dx(1)) ** 0.5
        w_l2_err_ = assemble((w - w_1) ** 2 * dx(2)) ** 0.5
        z_l2_err_ = assemble((z - z_1) ** 2 * dx(2)) ** 0.5

        u_l2_err.append(u_l2_err_)
        v_l2_err.append(v_l2_err_)
        w_l2_err.append(w_l2_err_)
        z_l2_err.append(z_l2_err_)

        # Check if total mass conservation law holds
        total_mass_femsol = assemble((u + v) * dx(1) + (w + z) * dx(2))

        u_1.assign(u)
        v_1.assign(v)
        w_1.assign(w)
        z_1.assign(z)

        vtkfile_u << (u_1, t)
        vtkfile_v << (v_1, t)
        vtkfile_w << (w_1, t)
        vtkfile_z << (z_1, t)
        time.append(t)

        print(
            "Time-step {} of {}: \t{:.15f}\t{:.15f}\t{:.15f}\t{:.15f}\t{:.15f}".format(
                n,
                num_steps,
                u_l2_err_,
                v_l2_err_,
                w_l2_err_,
                z_l2_err_,
                total_mass_femsol / total_mass_previous,
            )
        )

        # Update progress bar
        set_log_level(LogLevel.PROGRESS)
        progress += 1
        set_log_level(LogLevel.ERROR)

    data = {
        "u_l2_err": u_l2_err,
        "v_l2_err": v_l2_err,
        "w_l2_err": w_l2_err,
        "z_l2_err": z_l2_err,
        "time": time,
        "mesh_size": mesh.hmin(),
    }
    savemat("run{}/data/data_mesh{}.mat".format(ver, int(h ** (-1))), data)
    savemat(
        "run{}/data/phys_params_mesh{}.mat".format(ver, int(h ** (-1))), phys_params
    )

    plt.figure(figsize=(12, 8))
    plt.plot(u_l2_err, "r:*", label=r"$||u^{n} - u^{n-1}||_{L^2}$")
    plt.plot(v_l2_err, "b:*", label=r"$||v^{n} - v^{n-1}||_{L^2}$")
    plt.plot(w_l2_err, "g:*", label=r"$||w^{n} - w^{n-1}||_{L^2}$")
    plt.plot(z_l2_err, "m:*", label=r"$||z^{n} - z^{n-1}||_{L^2}$")
    plt.legend(loc="best", fontsize=20)
    plt.xlabel("iteration", fontsize=20)
    plt.ylabel(r"error in the $L^2$-norm", fontsize=20)
    plt.yscale("log")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.title("Error per iteration of the one-step Picard method", fontsize=25)
    plt.savefig("run{}/data/errorplot{}.png".format(ver, n), dpi=300)
    plt.close("all")
    return


if __name__ == "__main__":
    ver = "_nonlin_parabolic"

    # Create folders for saving run data
    if not os.path.exists("run{}/data".format(ver)):
        os.makedirs("run{}/data".format(ver))
    else:
        shutil.rmtree("run{}".format(ver))
        os.makedirs("run{}/data".format(ver))

    iters = 5  # max number of mesh refinements, if lim is not reached
    ratio = 0.5
    radius = 10
    phys_params = {
        "d3": 1.0,
        "d2": 1.0,
        "mu": 1.0,
        "nu": 1.0,
        "kappa": 1.0,
        "v_m": 1.0,
        "k_m": 1.0,
        "ctotal": 1.0,
    }

    t_final = 50.0
    num_steps = 500
    tol = 1e-8

    test_run(ver, iters, radius, ratio, phys_params, t_final, num_steps, tol)
