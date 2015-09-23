from memory_profiler import profile

import model_builder as mdb

@profile
def main1():
    dirs = [ x.rstrip("\n") for x in open("ticatemps","r").readlines() ]

    config = {"name":"1E0G","bead_repr":"CA",
              "pairwise_params_file":"%s/pairwise_params" % dirs[0],
              "model_params_file":"%s/dummy_model_params" % dirs[0]}
              #"model_params_file":"%s/model_params" % dirs[0]}

    modelopts = mdb.inputs._empty_model_opts()

    print "Options not shown default to None"
    mdb.inputs.load_model_section(config.items(),modelopts)
    mdb.inputs._add_pair_opts(modelopts)            # Pairwise params file
    modelopts["pdb"] = "%s.pdb" % modelopts["name"]

    #print modelopts
    model = mdb.models.CoarseGrainedModel(**modelopts)

@profile
def main():
    model,fitopts = mdb.inputs.load_model("1E0G")


if __name__ == "__main__":
    #main()
    model,fitopts = mdb.inputs.load_model("1E0G")
