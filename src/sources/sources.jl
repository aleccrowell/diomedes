"""
    DataSource

Supertype for data source adapters. A concrete source implements

    fetch_results(src, seasons) -> DataFrame

returning rows conforming to `RESULT_SCHEMA` (see `validate_results`).
"""
abstract type DataSource end

"""
    fetch_results(src::DataSource, seasons) -> DataFrame

Fetch and normalise results for `seasons` (an integer or collection of integers).
"""
function fetch_results end

fetch_results(src::DataSource, season::Integer) = fetch_results(src, [season])

# Adapters implement `fetch_results(src::MySource, seasons::AbstractVector{<:Integer})`.

# Helpers shared by adapters

"""
The Indianapolis 500 counted towards the F1 World Championship from 1950 to
1960, but it was effectively a separate series: different cars and a field of
drivers who mostly raced only there. The F1 adapters drop it by default.
"""
const INDY500 = "Indianapolis 500"

"`missing` for JSON null / absent keys, else the value converted to `String`."
maybestring(x) = (x === nothing || x === missing) ? missing : string(x)

"Parse an integer from a string/number, returning `missing` on failure."
maybeint(x::Integer) = Int(x)
maybeint(x::AbstractString) = something(tryparse(Int, x), missing)
maybeint(::Any) = missing

"Parse a date (Date, or \"yyyy-mm-dd...\" string), returning `missing` on failure."
maybedate(x::Date) = x
maybedate(x::AbstractString) = length(x) >= 10 ? something(tryparse(Date, x[1:10]), missing) : missing
maybedate(::Any) = missing

maybefloat(x::Real) = Float64(x)
maybefloat(x::AbstractString) = something(tryparse(Float64, x), missing)
maybefloat(::Any) = missing
