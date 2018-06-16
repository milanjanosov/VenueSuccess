


def get_inputs():

    inputs = {}
    for line in open('inputparams.dat'):
        fields = line.strip().split(',')
        inputs[fields[0]] = [float(fff) for fff in fields[1:]]

    return inputs
    
