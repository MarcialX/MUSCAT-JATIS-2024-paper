# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# msg_custom.py
# Custom messages
#
# Marcial Becerril, @ 15 January 2022
# Latest Revision: 15 Jan 2022, 18:45 GMT
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

# Defined colours
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def msg(text, type):
    if 'info' in type:
        print ( OKBLUE + 'INFO. ' + text + ENDC )
    elif 'ok' in type:
        print ( OKGREEN + text + ENDC )
    elif 'warn' in type:
        print ( WARNING + 'WARNING. ' + text + ENDC )
    elif 'fail' in type:
        print ( FAIL + 'ERROR. ' + text + ENDC )
    else:
        print ( FAIL + 'ERROR! Command not identified.' )
