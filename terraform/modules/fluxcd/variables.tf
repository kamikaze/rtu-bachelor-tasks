
variable "github_token" {
  sensitive = true
  type      = string
}

variable "github_org" {
  type = string
}

variable "github_repository" {
  type    = string
  default = "infra"
}

variable "public_key_openssh" {
  type = string
}

variable "git_infra_repo_url" {
  type     = string
  default  = ""
  nullable = false

  validation {
    condition     = length(var.git_infra_repo_url) >= 6
    error_message = "Git repository URL must be at least 6 characters long."
  }
}

variable "flux_git_infra_target_path" {
  type     = string
  default  = "fluxcd/clusters/dev"
  nullable = false

  validation {
    condition     = length(var.flux_git_infra_target_path) >= 3
    error_message = "Git target path must be at least 3 characters long."
  }
}

variable "flux_git_infra_branch" {
  type     = string
  default  = "main"
  nullable = false

  validation {
    condition     = length(var.flux_git_infra_branch) >= 1
    error_message = "Git branch name must be at least 1 characters long."
  }
}
