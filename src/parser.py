import re

class Parser:

    def __init__(self, path: str) -> None:
        with open(path, 'r') as f:
            self.source = f.read()

    def get_methods(self) -> tuple[list[str], list[str]]:
        lines = self.source.splitlines()
        methods = []
        names = []
        start = None
        brace_count = 0
        for idx, line in enumerate(lines):
            sig_match = re.match(r"^\s*(public|protected|private|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{", line)
            if sig_match:
                start = idx
                brace_count = line.count('{') - line.count('}')
                continue
            if start is not None:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    method_block = '\n'.join(lines[start:idx+1])
                    methods.append(method_block)
                    name = re.match(r".*\s+(\w+)\s*\([^)]*\).*", lines[start])
                    names.append(name.group(1) if name else None)
                    start = None
        return methods, names