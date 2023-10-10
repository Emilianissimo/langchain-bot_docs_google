from langchain.tools.base import ToolException


def tool_error_handler(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )
