import os

def conf_file_modify(pop):
    config_file = open(os.path.join("utils", "config-feedforward.txt"), "w")

    config_file.write("[NEAT]\n")
    config_file.write("fitness_criterion     = max\n")
    config_file.write("fitness_threshold     = 50\n")
    config_file.write("pop_size              = " + str(pop) + "\n")
    config_file.write("reset_on_extinction   = False\n\n")

    config_file.write("[DefaultGenome]\n")
    config_file.write("# node activation options\n")
    config_file.write("activation_default      = tanh\n")
    config_file.write("activation_mutate_rate  = 0.0\n")
    config_file.write("activation_options      = tanh\n\n")

    config_file.write("# node aggregation options\n")
    config_file.write("aggregation_default     = sum\n")
    config_file.write("aggregation_mutate_rate = 0.0\n")
    config_file.write("aggregation_options     = sum\n\n")

    config_file.write("# node bias options\n")
    config_file.write("bias_init_mean          = 0.0\n")
    config_file.write("bias_init_stdev         = 1.0\n")
    config_file.write("bias_max_value          = 30.0\n")
    config_file.write("bias_min_value          = -30.0\n")
    config_file.write("bias_mutate_power       = 0.5\n")
    config_file.write("bias_mutate_rate        = 0.7\n")
    config_file.write("bias_replace_rate       = 0.1\n\n")

    config_file.write("# genome compatibility options\n")
    config_file.write("compatibility_disjoint_coefficient = 1.0\n")
    config_file.write("compatibility_weight_coefficient   = 0.5\n\n")

    config_file.write("# connection add/remove rates\n")
    config_file.write("conn_add_prob           = 0.5\n")
    config_file.write("conn_delete_prob        = 0.5\n\n")

    config_file.write("# connection enable options\n")
    config_file.write("enabled_default         = True\n")
    config_file.write("enabled_mutate_rate     = 0.01\n\n")

    config_file.write("feed_forward            = True\n")
    config_file.write("initial_connection      = full\n\n")

    config_file.write("# node add/remove rates\n")
    config_file.write("node_add_prob           = 0.2\n")
    config_file.write("node_delete_prob        = 0.2\n\n")

    config_file.write("# network parameters\n")
    config_file.write("num_hidden              = 0\n")
    config_file.write("num_inputs              = 3\n")
    config_file.write("num_outputs             = 1\n\n")

    config_file.write("# node response options\n")
    config_file.write("response_init_mean      = 1.0\n")
    config_file.write("response_init_stdev     = 0.0\n")
    config_file.write("response_max_value      = 30.0\n")
    config_file.write("response_min_value      = -30.0\n")
    config_file.write("response_mutate_power   = 0.0\n")
    config_file.write("response_mutate_rate    = 0.0\n")
    config_file.write("response_replace_rate   = 0.0\n\n")

    config_file.write("# connection weight options\n")
    config_file.write("weight_init_mean        = 0.0\n")
    config_file.write("weight_init_stdev       = 1.0\n")
    config_file.write("weight_max_value        = 30\n")
    config_file.write("weight_min_value        = -30\n")
    config_file.write("weight_mutate_power     = 0.5\n")
    config_file.write("weight_mutate_rate      = 0.8\n")
    config_file.write("weight_replace_rate     = 0.1\n\n")

    config_file.write("[DefaultSpeciesSet]\n")
    config_file.write("compatibility_threshold = 3.0\n\n")

    config_file.write("[DefaultStagnation]\n")
    config_file.write("species_fitness_func = max\n")
    config_file.write("max_stagnation       = 20\n")
    config_file.write("species_elitism      = 2\n\n")

    config_file.write("[DefaultReproduction]\n")
    config_file.write("elitism            = 2\n")
    config_file.write("survival_threshold = 0.2\n")