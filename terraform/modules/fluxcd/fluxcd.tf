resource "github_repository_deploy_key" "this" {
  title      = "FluxCD"
  repository = var.github_repository
  key        = var.public_key_openssh
  read_only  = "false"
}

resource "flux_bootstrap_git" "this" {
  depends_on = [
    github_repository_deploy_key.this
  ]
  components_extra = ["image-reflector-controller", "image-automation-controller"]

  path = "fluxcd/clusters/dev"
}
