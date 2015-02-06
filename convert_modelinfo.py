import ConfigParser



if __name__ == "__main__":

    subdir = "SH3"

    config = ConfigParser.SafeConfigParser()
    config.add_section("model")
    config.add_section("fitting")


    info_file = open('%s/model.info' % subdir,'r')
    line = info_file.readline()
    inputs = {"Dry_Run":dry_run}
    while line != '':
        field = line.split()[1]
        value = info_file.readline().rstrip("\n")
        if field == "Reference":
            break
        elif field in ["Interaction_Groups","Model_Name",
                        "Backbone_params","Backbone_param_vals",
                        "Interaction_Types","Tf_iteration","Mut_iteration",
                        "Contact_Type","Contact_params","Contact_Energies"]:
            pass
        elif field == "Iteration":
            inputs[field] = int(value)
        elif field in ["N_Native_Contacts","N_Native_Pairs"]:
            field = "N_Native_Pairs"
            if value != "None":
                inputs[field] = int(value)
            else:
                inputs[field] = None
        elif field == "Fitting_Params":
            field = "Fitting_Params_File"
            inputs[field] = value
        else:
            inputs[field] = value
        line = info_file.readline()
