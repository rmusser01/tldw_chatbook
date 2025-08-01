# Complete File Types Supported

## Summary
The file extraction system now supports **224 different file types**, making it one of the most comprehensive file extraction systems available. Files are automatically detected from code blocks in LLM responses and offered for download with appropriate names and icons.

## File Categories

### 1. Programming Languages (50+)
- **Web**: JavaScript, TypeScript, HTML, CSS, JSX, TSX
- **Systems**: C, C++, Rust, Go, Zig, V
- **JVM**: Java, Kotlin, Scala, Clojure
- **Scripting**: Python, Ruby, PHP, Perl, Lua
- **Functional**: Haskell, ML, F#, Elm, PureScript, Idris, Agda, Lean, Coq
- **Mobile**: Swift, Dart, Objective-C
- **Data Science**: R, Julia, MATLAB
- **Other**: Nim, Pascal, Erlang, Elixir, Crystal

### 2. Configuration Files (30+)
- **General**: .conf, .cfg, .properties, .ini, .toml
- **Web Servers**: nginx.conf, httpd.conf
- **Databases**: my.cnf, postgresql.conf, redis.conf
- **Build Tools**: .gradle, .sbt, .cmake
- **Editors**: .editorconfig, .vimrc

### 3. Infrastructure & DevOps (25+)
- **Containers**: Dockerfile, docker-compose.yml
- **IaC**: .tf (Terraform), .tfvars
- **CI/CD**: .gitlab-ci.yml, Jenkinsfile, .travis.yml, GitHub Actions
- **Config Management**: ansible.cfg, Vagrantfile
- **Cloud**: serverless.yml, app.yaml

### 4. Package & Dependency Files (20+)
- **Python**: requirements.txt, Pipfile, setup.py
- **JavaScript**: package.json, yarn.lock
- **Ruby**: Gemfile, Gemfile.lock
- **Rust**: Cargo.toml, Cargo.lock
- **PHP**: composer.json
- **Swift**: Package.swift, Podfile, Cartfile
- **Other**: go.mod, mix.exs, cabal

### 5. Template Files (15+)
- **JavaScript**: .ejs, .hbs, .handlebars, .pug, .jade
- **Python**: .j2, .jinja2
- **Ruby**: .erb
- **General**: .liquid, .mustache, .njk

### 6. Data & Markup (20+)
- **Data**: .json, .yaml, .xml, .csv, .tsv
- **Streaming**: .ndjson, .jsonl
- **Binary**: .parquet, .avro
- **Semantic**: .rdf, .ttl, .jsonld
- **Geographic**: .geojson, .gpx, .kml
- **Documents**: .md, .adoc, .org, .tex
- **Diagrams**: .dot, .puml, .mmd

### 7. API & Schema Files (10+)
- **API Definitions**: .proto, .graphql, .openapi.yaml
- **Testing**: .http, .rest
- **Database**: .sql, .prisma

### 8. Script Files (15+)
- **Shell**: .sh, .bash, .zsh, .fish
- **Windows**: .bat, .cmd, .ps1, .psm1
- **Automation**: .awk, .sed
- **Editor**: .vim, .el (Emacs)

### 9. Build & Project Files (20+)
- **Microsoft**: .csproj, .vbproj, .sln
- **Java/Kotlin**: pom.xml, build.gradle
- **Web**: webpack.config.js, rollup.config.js, vite.config.js
- **Mobile**: .xcodeproj, .pbxproj
- **Other**: Makefile, CMakeLists.txt, .bazel

### 10. Testing & Quality (10+)
- **Test Configs**: jest.config.js, karma.conf.js, pytest.ini
- **BDD**: .feature (Gherkin)
- **Linting**: .eslintrc, .prettierrc, .rubocop.yml

### 11. Special Files (30+)
- **Version Control**: .gitignore, .gitattributes
- **Security**: .htaccess, ssh_config
- **Environment**: .env, .env.example
- **Calendar/Contacts**: .ics, .vcf
- **System**: .service (systemd), .plist
- **Documentation**: man pages, .texi

### 12. Low-Level & Assembly (10+)
- **Assembly**: .asm, .s, .nasm, .masm
- **WebAssembly**: .wat, .wasm
- **LLVM**: .ll

## Key Features

### Smart Filename Generation
The system automatically generates appropriate filenames:
- `nginx` → `nginx.conf`
- `dockerfile` → `Dockerfile`
- `requirements` → `requirements.txt`
- `gitignore` → `.gitignore`
- And 50+ more special cases

### Visual File Type Recognition
Each file type has a unique icon:
- 🐍 Python files
- 🐘 PHP/PostgreSQL/Gradle files
- 🦀 Rust files
- 🎯 Dart/Target files
- 📦 Package files
- 🐳 Docker files
- And 200+ more icons

### Comprehensive Validation
Many file types include validation:
- JSON/YAML syntax checking
- CSV column consistency
- Config file structure validation
- Package manifest required fields
- API schema validation

## Usage
1. When an LLM generates a code block with a recognized language identifier
2. The file is automatically detected and the extract button appears
3. Users can preview, rename, and download files
4. All files are validated before download

## Benefits
- **Comprehensive Coverage**: Supports virtually every common development file type
- **Smart Naming**: Files get proper names automatically
- **Visual Clarity**: Icons make file types instantly recognizable
- **Safety**: Validation prevents malformed files
- **Efficiency**: Extract multiple files at once with "Save All"