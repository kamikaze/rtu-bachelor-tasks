data "aws_eks_cluster_auth" "eks" {
  name = var.eks_cluster_name
}

resource "kubernetes_namespace" "terraform" {
  count = var.eks_enabled ? 1 : 0
  metadata {
    name = "terraform"
  }
}

resource "kubernetes_secret" "awssm_secret" {
  count = var.eks_enabled ? 1 : 0
  depends_on = [
    kubernetes_namespace.terraform[0]
  ]
  metadata {
    name      = "awssm-secret"
    namespace = kubernetes_namespace.terraform[0].metadata[0].name
  }

  data = {
    "access-key"        = var.aws_access_key_id
    "secret-access-key" = var.aws_access_key_secret
  }
}
