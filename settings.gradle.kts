rootProject.name = "ekg-gds-emr-2026"

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

include(
    "core-signal",
    "core-features",
    "rule-engine",
    "inference",
    "pipeline",
    "persistence",
    "app-fx",
)
