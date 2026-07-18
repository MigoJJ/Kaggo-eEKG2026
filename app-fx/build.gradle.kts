plugins {
    application
    alias(libs.plugins.javafx)
}

javafx {
    version = "25.0.1"
    modules = listOf("javafx.controls", "javafx.fxml", "javafx.swing")
}

dependencies {
    implementation(project(":pipeline"))
    implementation(project(":core-signal"))
    implementation(project(":core-features"))
    implementation(project(":rule-engine"))
    implementation(project(":persistence"))

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

application {
    mainClass.set("emr.ekg.app.EmrApp")
}

tasks.named<JavaExec>("run") {
    System.getProperty("ekg.debug.screenshot")?.let { systemProperty("ekg.debug.screenshot", it) }
}
