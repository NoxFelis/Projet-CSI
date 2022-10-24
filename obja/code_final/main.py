import boundary_curves as bc

DEFAULT_INPUT = "../sphere_input.obj"
DEFAULT_BASE = "../sphere_base.obj"
DEFAULT_NEW_INPUT = "../sphere_new_input.obj"

def main(args=None):

    patchs_limits, new_input_mesh_path, reel_indexes = bc.main(input_mesh_path=DEFAULT_INPUT, base_mesh_path=DEFAULT_BASE, new_input_mesh_path=DEFAULT_NEW_INPUT)

if __name__ == "__main__":
    main()
