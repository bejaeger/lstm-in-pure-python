# Simple class for logging output

SHOW_PREPROCESSING_OUTPUT = True

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    WHITE = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_color(type):
    if type == "error":
        color = bcolors.FAIL
    elif type == "warning":
        color = bcolors.WARNING
    else:
        color = bcolors.WHITE
    return color

def pp(text, type=None):
    if SHOW_PREPROCESSING_OUTPUT == False: return
    print(get_color(type) + "PREPROCESSING: {}".format(text) + bcolors.ENDC)

def info(text, type=None):
    if SHOW_PREPROCESSING_OUTPUT == False: return
    print(get_color(type) + "INFO: {}".format(text) + bcolors.ENDC)

def error(text):
    print(bcolors.FAIL + "INFO: {}".format(text) + bcolors.ENDC)

def sample(text):
    print(bcolors.HEADER + "-------------------" + bcolors.ENDC)
    print(bcolors.OKBLUE + "{}".format(text) + bcolors.ENDC)
    print(bcolors.HEADER + "-------------------" + bcolors.ENDC)

def header(text):
    if SHOW_PREPROCESSING_OUTPUT == False: return
    print(bcolors.HEADER + "-------------------" + bcolors.ENDC)
    print(bcolors.HEADER + "{}".format(text) + bcolors.ENDC)
    print(bcolors.HEADER + "-------------------" + bcolors.ENDC)

def train(text):
    print(bcolors.OKGREEN + "{}".format(text) + bcolors.ENDC)


