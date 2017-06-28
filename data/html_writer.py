class HtmlWriter(object):
    def __init__(self, result_file_path):
        self._result_file = open(result_file_path, mode='w')

        self.write("""
<html>
    <head>
        <style>
            .result_box{
                margin: 30px 5px;
                padding-left:10px;
            }

            .error{
                border-left: 5px solid red;
            }
        </style>
    </head>
<body>""")

    def write_result_box(self, items, extra_classes=''):
        if len(extra_classes) > 0:
            extra_classes = " " + extra_classes

        self.write("<div class='result_box" + extra_classes + "'>")
        for item in items:
            if isinstance(item, str):
                self.write(item + " ")
            else:
                text = None
                properties = ""
                for property in item:
                    value = item[property]

                    if property is "text":
                        text = value
                        continue

                    properties += " " + "data-" + property + "='" + str(value) + "'"

                self.write("<span " + properties + ">" + text + "</span> ")

        self.write("</div>")

    def h1(self, text):
        self.write("<h1>" + text + "</h1>")

    def write(self, text):
        self._result_file.write(text)

    def close(self):

        self.write("""
        <script>
            var spans = document.getElementsByTagName("span");

            for (var i=0, max=spans.length; i < max; i++) {
                span=spans[i];
                if(span.dataset.label){
                    span.style.color='blue';
                    span.style.fontWeight='bold';
                }

               span.style.backgroundColor='rgba(255,0,0,'+span.dataset.attention+')';
            }
        </script>
        """
                   )

        self.write("</body></html>")
        self._result_file.close()
