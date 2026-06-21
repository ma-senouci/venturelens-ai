class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSidebarContext:
    def __init__(self, parent):
        self._parent = parent

    def __enter__(self):
        self._parent._in_sidebar = True
        return self

    def __exit__(self, *args):
        self._parent._in_sidebar = False
        return False

    def header(self, text, *args, **kwargs):
        self._parent.sidebar_headers.append(text)

    def caption(self, text, *args, **kwargs):
        self._parent.sidebar_captions.append(text)


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.captions: list[str] = []
        self.subheaders: list[str] = []
        self.markdowns: list[str] = []
        self.expander_labels: list[str] = []
        self.sidebar_headers: list[str] = []
        self.sidebar_captions: list[str] = []
        self.sidebar_buttons: list[dict] = []
        self.sidebar_text_inputs: list[dict] = []
        self.form_submit_buttons: list[dict] = []
        self.buttons: list[dict] = []
        self._in_sidebar = False
        self._sidebar_ctx = _FakeSidebarContext(self)

    @property
    def sidebar(self):
        return self._sidebar_ctx

    def set_page_config(self, **kwargs):
        pass

    def error(self, message):
        self.errors.append(message)

    def stop(self):
        raise AssertionError("st.stop should not be called")

    def title(self, *args, **kwargs):
        pass

    def subheader(self, text, *args, **kwargs):
        self.subheaders.append(text)

    def markdown(self, text, *args, **kwargs):
        self.markdowns.append(text)

    def text(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass

    def form(self, *args, **kwargs):
        return _NullContext()

    def text_input(self, label="", *args, **kwargs):
        entry = {"label": label, "type": kwargs.get("type"), "key": kwargs.get("key")}
        if self._in_sidebar:
            self.sidebar_text_inputs.append(entry)
        key = kwargs.get("key")
        return self.session_state.get(key, "") if key else ""

    def text_area(self, *args, **kwargs):
        return ""

    def form_submit_button(self, label="", *args, **kwargs):
        self.form_submit_buttons.append({"label": label, "disabled": kwargs.get("disabled", False)})
        return False

    def button(self, label="", *args, **kwargs):
        if self._in_sidebar:
            self.sidebar_buttons.append(
                {
                    "label": label,
                    "key": kwargs.get("key"),
                    "disabled": kwargs.get("disabled", False),
                }
            )
            return False
        self.buttons.append({"label": label, "disabled": kwargs.get("disabled", False)})
        return False

    def status(self, *args, **kwargs):
        return _NullContext()

    def write(self, *args, **kwargs):
        pass

    def caption(self, message, *args, **kwargs):
        if self._in_sidebar:
            self.sidebar_captions.append(message)
        else:
            self.captions.append(message)

    def header(self, text, *args, **kwargs):
        if self._in_sidebar:
            self.sidebar_headers.append(text)

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)

    def info(self, message, *args, **kwargs):
        self.infos.append(message)

    def expander(self, label, *args, **kwargs):
        self.expander_labels.append(label)
        return _NullContext()

    def rerun(self):
        pass

    def _clear_sidebar(self):
        self.sidebar_headers.clear()
        self.sidebar_captions.clear()
        self.sidebar_buttons.clear()
