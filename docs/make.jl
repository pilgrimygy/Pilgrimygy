using Documenter, Example

include("common.jl")

recursive_replace(@__DIR__, r"\.zh(?<ext>(\.md)?)$" => s"\g<ext>")

makedocs(
    format = Documenter.HTML(
        prettyurls = true,
        analytics = "G-5NBJ9X617X",
        lang = "zh-CN",
        footer = "本站基于 [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) 和 [Julia 编程语言](https://julialang.org/) 构建，所有内容默认遵循[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)协议。",
        assets = [
            "assets/custom.css",
        ]
    ),
    sitename="pilgrim",
    pages = [
        "👋 关于" => "index.md",
        "🔗 友链" => "blogroll.md",
        "🗃️ 研究" => [

        ],
        "💻 技术" => [

        ],
        "📚 杂谈" => [

        ]
    ]
)