def default_find_error(files):
    import re
    errors = ''

    for file_name in files:
        with open(file_name) as f:
            num = 1
            for line in f:
                if re.search(r'(?i)error', line):
                    errors += file_name + " " + str(num) + " " + line
                if 'Traceback (most recent call last)' in line:
                    errors += file_name + " " + str(num) + " " + line

                num += 1

    return errors

