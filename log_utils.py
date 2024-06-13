from prettytable import PrettyTable

from colorama import Fore, Back, Style
import numpy as np
import _global


class Logger:

    def __init__(self) -> None:
        pass


def print_error():

    pass


def print_stereo_camera_data_info(cam_01, min_t, max_t, len_t):

    VERBOSE = _global.get_value('verbose')
    if VERBOSE:
        table = PrettyTable(
            ["cam-id", "time(min/ms)", "time(max/ms)", "length"])
        table.add_row([cam_01, min_t, max_t, len_t])
        print(Fore.GREEN, table, Style.RESET_ALL)


def print_info_green(ss, output=False):
    VERBOSE = _global.get_value('verbose')
    if VERBOSE or output:
        print(Fore.GREEN, ss, Style.RESET_ALL)


def print_info_red(ss, output=False):
    VERBOSE = _global.get_value('verbose')
    if VERBOSE or output:
        print(Fore.RED, ss, Style.RESET_ALL)


def print_info_blue(ss, output=False):
    VERBOSE = _global.get_value('verbose')
    if VERBOSE or output:
        print(Fore.LIGHTBLUE_EX, ss, Style.RESET_ALL)


def print_pts_refine_error(cam01, errors, output=False):
    VERBOSE = _global.get_value('verbose')
    if VERBOSE or output:
        print_info_blue("The 3d point refine error:")
        table = PrettyTable()
        table.add_column("cam-id-iters", [f"{cam01}"])
        for i in range(len(errors)):
            table.add_column(f"iter-{i}", [errors[i]])
        print(Fore.GREEN, table, Style.RESET_ALL)


def print_multi_errors(errors, output=False):
    VERBOSE = _global.get_value('verbose')

    if VERBOSE or output:
        mean_error = errors.mean(1).mean()

        if mean_error > 5.0:
            mean_error = Fore.RED+f"{mean_error}"
        else:
            mean_error = Fore.GREEN+f"{mean_error}"
        print_info_blue(f"The final camera mean errors: {mean_error}")
        table = PrettyTable()
        cams = []
        cam_num = errors.shape[0]
        for i in range(cam_num):
            cam_name = Fore.GREEN+f"cam{i}"
            cams.append(cam_name)
        table.add_column("cam-names", cams)

        # errors: [n,n]
        index = np.where(errors > 5.0)
        index1 = np.where(errors <= 5.0)
        errors = errors.tolist()
        for i, j in zip(index[0], index[1]):
            errors[i][j] = Fore.RED+f"{errors[i][j]}"+Style.RESET_ALL
        for i, j in zip(index1[0], index1[1]):
            errors[i][j] = Fore.GREEN+f"{errors[i][j]}"  # +Style.RESET_ALL

        for i in range(cam_num):
            table.add_column(f"cam{i}", errors[i])

        print(Fore.GREEN, table, Style.RESET_ALL)


def print_pts_frame(cams, num_frames, output=False):
    VERBOSE = _global.get_value('verbose')
    if VERBOSE or output:
        table = PrettyTable(cams)
        table.add_row(num_frames)
        print(Fore.GREEN, table, Style.RESET_ALL)


# table = PrettyTable(["name", "score"])
# table.add_row(["Bob", 67])
# table.add_row(["grizzly", 45])
# table.add_row(["Tom of Caerbannog", 78])
# table.add_row(["cat", 34])
# table.add_row(["Tony", 39])
# table.add_row(["dolphin", 45])
# table.add_row(["albatross", 24])
# table.sort_key("name")
# table.reversesort = True


# print(Fore.RED + "some red text")
# print(Back.GREEN + "and with a green background")
# print(Style.DIM + "and in dim text")
# print(Style.RESET_ALL)
# print("back to normal now!!")


# print(Fore.BLACK + 'BLACK')
# print(Fore.BLUE + 'BLUE')
# print(Fore.CYAN + 'CYAN')
# print(Fore.GREEN + 'GREEN')
# print(Fore.LIGHTBLACK_EX + 'LIGHTBLACK_EX')
# print(Fore.LIGHTBLUE_EX + 'LIGHTBLUE_EX')
# print(Fore.LIGHTCYAN_EX + 'LIGHTCYAN_EX')
# print(Fore.LIGHTGREEN_EX + 'LIGHTGREEN_EX')
# print(Fore.LIGHTMAGENTA_EX + 'LIGHTMAGENTA_EX')
# print(Fore.LIGHTRED_EX + 'LIGHTRED_EX')
# print(Fore.LIGHTWHITE_EX + 'LIGHTWHITE_EX')
# print(Fore.LIGHTYELLOW_EX + 'LIGHTYELLOW_EX')
# print(Fore.MAGENTA + 'MAGENTA')
# print(Fore.RED + 'RED')
# print(Fore.RESET + 'RESET')
# print(Fore.WHITE + 'WHITE')
# print(Fore.YELLOW + 'YELLOW')
