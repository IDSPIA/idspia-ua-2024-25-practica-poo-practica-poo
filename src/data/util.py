class Util:
    @staticmethod
    def parse_line(line: str, separator: str = " "):
        try:
            data = [float(v) for v in line.split(separator)]
        except ValueError as e:
            print("Couldn't properly load the data")
            # What are the conditions for failure (i.e just non numerical data or also too much or too little data
            return None
        else:
            return data
