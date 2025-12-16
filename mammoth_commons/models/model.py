class Model:
    integration = "dsl.Model"

    def __init__(self):
        self.description = ""

    def to_description(self):
        if not self.description:
            return ""
        if isinstance(self.description, str):
            desc = self.description.split("Args:")[0] + "<br>"
        elif isinstance(self.description, dict):
            desc = ""
            for key, value in self.description.items():
                desc += f"<h3>{key}</h3>" + value.replace("\n", "<br>") + "<br>"
        else:
            raise Exception("Model description must be a string or a dictionary.")

        from mammoth_commons.exports import get_description_header

        header = get_description_header(desc)
        return header + desc.replace(header, "", 1)
