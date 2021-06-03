import libcst as cst


def is_correct(x):
    try:
        cst.parse_module(x)
    except Exception as e:
        # print(e)
        return False
    return True


if __name__ == "__main__":
    import sys
    if sys.argv[1].endswith('.py'):
        with open(sys.argv[1], 'r') as f:
            content = f.read()
            print(is_correct(content))
    else:
        print(is_correct(sys.argv[1]))
